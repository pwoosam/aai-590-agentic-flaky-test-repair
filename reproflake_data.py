"""ReproFlake acquisition, repair-pair inspection, and leakage-aware partitioning."""

from __future__ import annotations

import hashlib
import csv
import codecs
import io
import json
import logging
import random
import re
import shutil
import stat
import subprocess
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
import libarchive
from unidiff import PatchSet
from unidiff.errors import UnidiffParseError

LOGGER = logging.getLogger(__name__)
TIMEOUT = (20, 120)
ARTIFACT = "https://anonymous.4open.science/api/repo/ReproFlake-C9E6"
RECORDS = (18474558, 18634501, 18605131)
METADATA = {
    "config.csv": "test_config.csv",
    "jira.csv": "research-data/Reproducible_JIRA_info.csv",
    "idoft.csv": "research-data/Reproducible_iDoFT_info.csv",
    "issues.csv": "issue_reports_updated.csv",
}
CATEGORIES = {
    "id": "Implementation-dependent",
    "od": "Order-dependent",
    "nio": "Non-idempotent outcome",
    "td": "Time-dependent",
    "britle": "Order-dependent",
    "unclassified": "Unclassified",
}


def session() -> requests.Session:
    client = requests.Session()
    client.headers["User-Agent"] = "ReproFlake-Capstone-EDA/1.0"
    return client


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    try:
        temporary.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def request_bytes(client: requests.Session, url: str, limit: int = 5_000_000) -> bytes:
    for attempt in range(3):
        with client.get(url, stream=True, timeout=TIMEOUT) as response:
            if response.status_code in {429, 500, 502, 503, 504} and attempt < 2:
                delay = response.headers.get("Retry-After", "5")
                seconds = (
                    float(delay)
                    if delay.isdigit()
                    else (
                        parsedate_to_datetime(delay) - datetime.now(timezone.utc)
                    ).total_seconds()
                )
                if seconds > 120:
                    response.raise_for_status()
                LOGGER.warning("HTTP %s; retrying %s", response.status_code, url)
                time.sleep(max(seconds, 0))
                continue
            response.raise_for_status()
            payload = bytearray()
            for block in response.iter_content(65536):
                if len(payload) + len(block) > limit:
                    raise ValueError(f"Response exceeds {limit:,}-byte limit: {url}")
                payload.extend(block)
            return bytes(payload)
    raise RuntimeError(f"Request retries exhausted: {url}")


def acquire_metadata(client: requests.Session, directory: Path) -> dict:
    """Cache source bytes and hashes; refuse silent reuse of modified local files."""
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "provenance.json"
    manifest = (
        json.loads(manifest_path.read_text())
        if manifest_path.exists()
        else {"files": {}}
    )
    sources = {name: f"{ARTIFACT}/file/{remote}" for name, remote in METADATA.items()}
    sources.update(
        {
            f"zenodo_{rid}.json": f"https://zenodo.org/api/records/{rid}"
            for rid in (*RECORDS, 18451345)
        }
    )
    for name, url in sources.items():
        path = directory / name
        cached = manifest["files"].get(name)
        if path.exists():
            if (
                not cached
                or cached["url"] != url
                or cached["sha256"] != sha256(path.read_bytes())
            ):
                raise ValueError(f"Unverified or modified metadata cache: {path}")
        else:
            payload = request_bytes(client, url)
            path.write_bytes(payload)
            manifest["files"][name] = {
                "url": url,
                "sha256": sha256(payload),
                "bytes": len(payload),
                "retrieved_utc": datetime.now(timezone.utc).isoformat(),
            }
            write_json(manifest_path, manifest)
    return manifest


def canonical_project(value: str) -> str:
    parsed = urlparse(value.strip().removesuffix(".git").rstrip("/"))
    parts = parsed.path.strip("/").split("/")
    if parts in (
        ["apache", "apache", "curator"],
        ["apache", "apache", "apex-core"],
        ["apache", "apache", "storm"],
    ):
        parts = parts[1:]
    if (
        parsed.scheme != "https"
        or parsed.netloc.lower() != "github.com"
        or len(parts) != 2
    ):
        raise ValueError(f"Unexpected project URL: {value}")
    return "/".join(parts).lower()


def load_metadata(
    directory: Path, audit_path: Path | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config = pd.read_csv(directory / "config.csv", dtype="string", encoding="utf-8-sig")
    issue = pd.read_csv(directory / "issues.csv", dtype="string", encoding="utf-8-sig")
    info = []
    for source in ("jira", "idoft"):
        table = pd.read_csv(
            directory / f"{source}.csv", dtype="string", encoding="utf-8-sig"
        )
        table.columns = table.columns.str.strip()
        table = table.rename(
            columns={
                "Container Name": "case_id",
                "Project (Github Link)": "project_url",
                "Flaky commit SHA": "flaky_commit",
                "Fixed commit SHA": "fixed_commit",
                "Zip name": "info_archive",
            }
        )
        table["discovery_source"] = source
        info.append(
            table[
                [
                    "case_id",
                    "project_url",
                    "flaky_commit",
                    "fixed_commit",
                    "info_archive",
                    "discovery_source",
                ]
            ]
        )
    info_frame = pd.concat(info, ignore_index=True)
    config = config.rename(
        columns={
            "result_container": "case_id",
            "zip": "archive_id",
            "polluter/state setter": "polluter",
            "flaky_test": "test_name",
            "nondexSeed": "nondex_seed",
            "java": "java_version",
            "url": "archive_url",
            "config": "execution_config",
        }
    )
    if config.case_id.isna().any() or config.case_id.duplicated().any():
        raise ValueError("Missing or duplicate test identities in configuration.")
    if set(config.case_id) != set(info_frame.case_id):
        raise ValueError(
            "Configuration and project provenance have different case identities."
        )
    joined = config.merge(
        info_frame, on="case_id", validate="one_to_one", indicator=True
    )
    if (
        not joined["_merge"].eq("both").all()
        or not joined.archive_id.eq(joined.info_archive).all()
    ):
        raise ValueError("Unmatched provenance or conflicting archive identity.")
    issue = issue.rename(
        columns={
            "result_container": "case_id",
            "PR Link": "fix_pr_url",
            "Issue Report": "issue_url",
        }
    )
    joined = joined.drop(columns=["_merge", "info_archive"]).merge(
        issue[["case_id", "fix_pr_url", "issue_url"]],
        on="case_id",
        how="left",
        validate="one_to_one",
    )
    issues = []
    details = []

    def record_changes(
        field: str,
        before: pd.Series,
        after: pd.Series,
        mask: pd.Series,
        action: str,
        rationale: str,
    ) -> None:
        for index in joined.index[mask.fillna(False)]:
            details.append(
                {
                    "case_id": joined.at[index, "case_id"],
                    "field": field,
                    "before": before.at[index],
                    "after": after.at[index],
                    "action": action,
                    "rationale": rationale,
                }
            )

    for column in joined:
        if pd.api.types.is_string_dtype(joined[column]):
            stripped = joined[column].str.strip().replace("", pd.NA)
            changed = joined[column].fillna("") != stripped.fillna("")
            count = int(changed.sum())
            if count:
                issues.append(
                    {"issue": "Whitespace normalized", "field": column, "count": count}
                )
                record_changes(
                    column,
                    joined[column],
                    stripped,
                    changed,
                    "Trim whitespace",
                    "Remove formatting artifacts before identity and type validation.",
                )
            joined[column] = stripped
    category = joined.test_type.str.lower()
    if not category.isin(CATEGORIES).all():
        raise ValueError(
            f"Unrecognized test types: {joined.loc[~category.isin(CATEGORIES), 'test_type'].unique()}"
        )
    joined["category"] = category.map(CATEGORIES).astype("string")
    record_changes(
        "category",
        joined.test_type,
        joined.category,
        category.isin(["unclassified", "britle"]),
        "Derive canonical mechanism",
        "Harmonize category spellings in a derived field; retain the raw annotation.",
    )
    issues.append(
        {
            "issue": "Unclassified spelling variants",
            "field": "test_type",
            "count": int(category.eq("unclassified").sum()),
        }
    )
    issues.append(
        {
            "issue": "Brittle spelling mapped to order-dependence; raw type retained",
            "field": "test_type",
            "count": int(category.eq("britle").sum()),
        }
    )
    for column in ("iterations", "java_version"):
        converted = pd.to_numeric(joined[column], errors="coerce")
        invalid = converted.isna() | converted.le(0) | converted.mod(1).ne(0)
        issues.append(
            {
                "issue": "Invalid positive integer",
                "field": column,
                "count": int(invalid.sum()),
            }
        )
        record_changes(
            column,
            joined[column],
            converted.mask(invalid),
            invalid & joined[column].notna(),
            "Set invalid measurement missing",
            "Reproduction settings require positive integers; do not invent replacements.",
        )
        joined[column] = converted.mask(invalid).astype("Int64")
    for column in ("flaky_commit", "fixed_commit"):
        valid = joined[column].str.fullmatch(r"[0-9a-fA-F]{40}", na=False)
        invalid = joined[column].notna() & ~valid
        issues.append(
            {
                "issue": "Invalid commit identifier",
                "field": column,
                "count": int(invalid.sum()),
            }
        )
        record_changes(
            column,
            joined[column],
            joined[column].where(valid),
            invalid,
            "Set invalid identifier missing",
            "A revision must be a complete 40-character hexadecimal identifier.",
        )
        joined[column] = joined[column].where(valid)
    joined["project"] = joined.project_url.map(canonical_project)
    record_changes(
        "project",
        joined.project_url,
        joined.project,
        joined.project_url.str.contains("/apache/apache/", regex=False),
        "Repair canonical project identity",
        "Correct the documented duplicated owner segment; preserve the original URL.",
    )
    issues.append(
        {
            "issue": "Duplicated apache owner segment corrected; raw URL retained",
            "field": "project_url",
            "count": int(
                joined.project_url.str.contains("/apache/apache/", regex=False).sum()
            ),
        }
    )
    joined["has_issue_reference"] = joined.issue_url.notna()
    joined["has_fix_pr_reference"] = joined.fix_pr_url.notna()
    joined["has_commit_pair"] = (
        joined.flaky_commit.notna() & joined.fixed_commit.notna()
    )
    joined["has_polluter"] = joined.polluter.notna()
    joined["has_seed"] = joined.nondex_seed.notna()
    joined["tests_per_archive"] = joined.groupby("archive_id").case_id.transform("size")
    for archive_id, group in joined.groupby("archive_id"):
        if group.project.nunique() != 1 or group.archive_url.nunique() != 1:
            raise ValueError(f"Ambiguous project/archive identity: {archive_id}")
    if audit_path is not None:
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            details,
            columns=["case_id", "field", "before", "after", "action", "rationale"],
        ).to_csv(audit_path, index=False)
    return joined, pd.DataFrame(issues), config


def archive_catalog(directory: Path) -> pd.DataFrame:
    rows = []
    for rid in RECORDS:
        record = json.loads((directory / f"zenodo_{rid}.json").read_text())
        for item in record["files"]:
            rows.append(
                {
                    "archive_id": item["key"].removesuffix(".zip"),
                    "file_name": item["key"],
                    "record": rid,
                    "url": item["links"]["self"],
                    "archive_bytes": item["size"],
                    "publisher_checksum": item["checksum"],
                    "license": record["metadata"]["license"]["id"],
                }
            )
    frame = pd.DataFrame(rows)
    if frame.archive_id.duplicated().any():
        raise ValueError("Archive identifiers are not unique across pinned records.")
    return frame


def file_hashes(path: Path) -> dict[str, str]:
    md5 = hashlib.md5(usedforsecurity=False)
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            md5.update(block)
            sha.update(block)
    return {"md5": md5.hexdigest(), "sha256": sha.hexdigest()}


def verify_download(path: Path, item: dict) -> dict:
    if path.stat().st_size != int(item["archive_bytes"]):
        raise ValueError(f"Incorrect downloaded size: {path.name}")
    digests = file_hashes(path)
    if "md5:" + digests["md5"] != item["publisher_checksum"]:
        raise ValueError(f"Publisher checksum mismatch: {path.name}")
    return {
        **item,
        **digests,
        "local_path": str(path),
        "verified_utc": datetime.now(timezone.utc).isoformat(),
    }


def download_archive(client: requests.Session, item: dict, directory: Path) -> dict:
    """Resume interrupted downloads, then verify every byte against the publisher."""
    directory.mkdir(parents=True, exist_ok=True)
    name = item["file_name"]
    if Path(name).name != name or name in {"", ".", ".."}:
        raise ValueError(f"Unsafe download filename: {name}")
    destination = directory / name
    if destination.exists():
        return verify_download(destination, item)
    partial = directory / (name + ".part")
    expected_size = int(item["archive_bytes"])
    for attempt in range(5):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > expected_size:
            raise ValueError(f"Oversized partial download: {name}")
        if offset == expected_size:
            result = verify_download(partial, item)
            partial.replace(destination)
            return {**result, "local_path": str(destination)}
        if shutil.disk_usage(directory).free < expected_size - offset + 256_000_000:
            raise OSError(f"Insufficient disk space for {name}")
        headers = {"Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            with client.get(
                item["url"], headers=headers, stream=True, timeout=TIMEOUT
            ) as response:
                if response.status_code in {429, 500, 502, 503, 504} and attempt < 4:
                    retry = response.headers.get("Retry-After", "10")
                    delay = (
                        float(retry)
                        if retry.isdigit()
                        else (
                            parsedate_to_datetime(retry) - datetime.now(timezone.utc)
                        ).total_seconds()
                    )
                    if delay > 120:
                        response.raise_for_status()
                    LOGGER.warning(
                        "Retrying %s after HTTP %s", name, response.status_code
                    )
                    time.sleep(max(delay, 0) + 1)
                    continue
                response.raise_for_status()
                if offset:
                    if (
                        response.status_code != 206
                        or response.headers.get("Content-Range")
                        != f"bytes {offset}-{expected_size - 1}/{expected_size}"
                    ):
                        raise ValueError(
                            f"Server did not honor exact resume range: {name}"
                        )
                elif response.status_code != 200:
                    raise ValueError(
                        f"Unexpected full-download status: {response.status_code}"
                    )
                if response.headers.get("Content-Encoding", "identity") != "identity":
                    raise ValueError("Encoded downloads cannot be resumed reliably.")
                received = offset
                with partial.open("ab" if offset else "wb") as stream:
                    for block in response.iter_content(1024 * 1024):
                        if received + len(block) > expected_size:
                            raise ValueError(f"Response exceeds published size: {name}")
                        stream.write(block)
                        received += len(block)
            if received != expected_size:
                raise requests.ConnectionError(f"Truncated download: {name}")
        except (
            requests.ConnectionError,
            requests.Timeout,
            requests.exceptions.ChunkedEncodingError,
        ):
            if attempt == 4:
                raise
            LOGGER.warning("Interrupted download; resuming %s", name)
            time.sleep(2**attempt)
            continue
        result = verify_download(partial, item)
        partial.replace(destination)
        return {**result, "local_path": str(destination)}
    raise RuntimeError(f"Download retries exhausted: {name}")


def download_corpus(
    catalog: pd.DataFrame, metadata: Path, directory: Path, workers: int = 4
) -> pd.DataFrame:
    if workers not in {1, 2, 3, 4}:
        raise ValueError("Use at most four concurrent archive downloads.")
    record = json.loads((metadata / "zenodo_18451345.json").read_text())
    item = next(f for f in record["files"] if f["key"] == "result.zip")
    results = {
        "archive_id": "__execution_results__",
        "file_name": item["key"],
        "record": 18451345,
        "url": item["links"]["self"],
        "archive_bytes": item["size"],
        "publisher_checksum": item["checksum"],
        "license": record["metadata"]["license"]["id"],
    }
    items = [results] + catalog.to_dict("records")
    directory.mkdir(parents=True, exist_ok=True)
    required = sum(
        max(
            int(i["archive_bytes"])
            - (
                (directory / (i["file_name"] + ".part")).stat().st_size
                if (directory / (i["file_name"] + ".part")).exists()
                else 0
            ),
            0,
        )
        for i in items
        if not (directory / i["file_name"]).exists()
    )
    if shutil.disk_usage(directory).free < required + 1_000_000_000:
        raise OSError(
            f"Full acquisition requires {required / 1e9:.2f} GB plus 1 GB margin."
        )
    records = []
    manifest_path = directory / "download_manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    completed = {item["file_name"]: item for item in previous}

    def retrieve(item: dict) -> dict:
        with session() as client:
            return download_archive(client, item, directory)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(retrieve, item): item for item in items}
        for future in as_completed(futures):
            verified = future.result()
            completed[verified["file_name"]] = verified
            records.append(verified)
            write_json(manifest_path, list(completed.values()))
            LOGGER.info(
                "Verified %d/%d archives: %s",
                len(records),
                len(items),
                verified["file_name"],
            )
    return pd.DataFrame(records).sort_values("file_name").reset_index(drop=True)


def project_split(frame: pd.DataFrame, seed: int = 590) -> pd.DataFrame:
    projects = sorted(frame.project.unique())
    if len(projects) < 3:
        raise ValueError(
            "At least three projects are required for independent partitions."
        )
    random.Random(seed).shuffle(projects)
    train_end = max(1, min(len(projects) - 2, round(0.7 * len(projects))))
    val_end = max(
        train_end + 1, min(len(projects) - 1, train_end + round(0.15 * len(projects)))
    )
    mapping = {
        project: "train" if i < train_end else "validation" if i < val_end else "test"
        for i, project in enumerate(projects)
    }
    result = frame.copy()
    result["split"] = result.project.map(mapping)
    if result.groupby("project").split.nunique().max() != 1:
        raise AssertionError("Project split leakage.")
    return result


def select_archives(cases: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    """One archive per randomly ordered project before selecting further versions."""
    if count <= 0:
        raise ValueError("Artifact sample size must be positive.")
    rng = random.Random(f"artifact-selection:{seed}")
    projects = sorted(cases.project.unique())
    rng.shuffle(projects)
    queues = {}
    for project in projects:
        archives = sorted(cases.loc[cases.project.eq(project), "archive_id"].unique())
        rng.shuffle(archives)
        queues[project] = archives
    chosen = []
    while len(chosen) < min(count, cases.archive_id.nunique()):
        for project in projects:
            if queues[project]:
                chosen.append(queues[project].pop())
                if len(chosen) == min(count, cases.archive_id.nunique()):
                    break
    return (
        cases.loc[cases.archive_id.isin(chosen)]
        .drop_duplicates("archive_id")
        .set_index("archive_id")
        .loc[chosen]
        .reset_index()
    )


class BudgetExceeded(RuntimeError):
    """Resource limit reached without requesting the full archive."""


@dataclass
class TransferBudget:
    limit: int
    used: int = 0

    def __post_init__(self) -> None:
        if self.limit <= 0:
            raise ValueError("Transfer limit must be positive.")


class RangeReader(io.RawIOBase):
    def __init__(
        self, client: requests.Session, url: str, size: int, budget: TransferBudget
    ):
        super().__init__()
        self.client, self.url, self.size, self.budget = client, url, size, budget
        self.position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence not in {io.SEEK_SET, io.SEEK_CUR, io.SEEK_END}:
            raise ValueError("Invalid seek mode.")
        position = offset + (
            0
            if whence == io.SEEK_SET
            else self.position if whence == io.SEEK_CUR else self.size
        )
        if position < 0:
            raise ValueError("Negative seek position.")
        self.position = position
        return position

    def read(self, size: int = -1) -> bytes:
        size = min(max(self.size - self.position, 0), size if size >= 0 else self.size)
        if not size:
            return b""
        if self.budget.used + size > self.budget.limit:
            raise BudgetExceeded(
                "Cumulative archive transfer cap reached (including ZIP indexes)."
            )
        start, end = self.position, self.position + size - 1
        for attempt in range(4):
            response = self.client.get(
                self.url,
                headers={
                    "Range": f"bytes={start}-{end}",
                    "Accept-Encoding": "identity",
                },
                stream=True,
                timeout=TIMEOUT,
            )
            if response.status_code in {429, 500, 502, 503, 504} and attempt < 3:
                delay = response.headers.get("Retry-After", "60")
                seconds = (
                    float(delay)
                    if delay.isdigit()
                    else (
                        parsedate_to_datetime(delay) - datetime.now(timezone.utc)
                    ).total_seconds()
                )
                if seconds > 120:
                    response.raise_for_status()
                response.close()
                LOGGER.info(
                    "Publisher throttled byte ranges; waiting %.0f seconds.", seconds
                )
                time.sleep(max(seconds, 0) + 1)
                continue
            break
        return self._consume(response, size, start, end)

    def _consume(
        self, response: requests.Response, size: int, start: int, end: int
    ) -> bytes:
        with response:
            response.raise_for_status()
            if response.status_code != 206:
                raise ValueError(
                    f"HTTP {response.status_code}: byte ranges required; full-download fallback forbidden."
                )
            if (
                response.headers.get("Content-Range")
                != f"bytes {start}-{end}/{self.size}"
            ):
                raise ValueError("Unexpected Content-Range.")
            if response.headers.get("Content-Encoding", "identity") != "identity":
                raise ValueError("Encoded byte ranges are unsupported.")
            chunks, received = [], 0
            for chunk in response.iter_content(min(size, 65536)):
                self.budget.used += len(chunk)
                received += len(chunk)
                if received > size or self.budget.used > self.budget.limit:
                    raise BudgetExceeded("Server exceeded the requested range.")
                chunks.append(chunk)
            if received != size:
                raise ValueError("Truncated byte range.")
            if int(response.headers.get("X-RateLimit-Remaining", "100")) <= 2:
                LOGGER.info(
                    "Publisher rate limit nearly exhausted; pausing range acquisition."
                )
                time.sleep(61)
        self.position += size
        return b"".join(chunks)


def safe_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or ".git" in path.parts
        or "\\" in value
        or "\x00" in value
        or re.match(r"^[A-Za-z]:", value)
    ):
        raise ValueError(f"Unsafe archive/patch path: {value!r}")
    return str(path)


def patch_path(value: str) -> str:
    if value == "/dev/null":
        return value
    for prefix in ("Flaky/", "Fixed/", "a/", "b/"):
        if value.startswith(prefix):
            return safe_path(value[len(prefix) :])
    raise ValueError(f"Unsupported patch path prefix: {value}")


def bounded_member(
    archive: zipfile.ZipFile, member: zipfile.ZipInfo, limit: int = 16_000_000
) -> bytes:
    safe_path(member.filename)
    if stat.S_ISLNK(member.external_attr >> 16):
        raise ValueError(f"Symlink member forbidden: {member.filename}")
    if member.file_size > limit or member.compress_size > limit:
        raise BudgetExceeded(f"Member exceeds {limit:,}-byte limit: {member.filename}")
    return archive.read(member)  # zipfile verifies the selected member's CRC.


def inspect_archive(
    client: requests.Session | None,
    item: dict,
    budget: TransferBudget | None,
    local_path: Path | None = None,
) -> dict:
    with ExitStack() as stack:
        if local_path is not None:
            resource = local_path
        elif client is not None and budget is not None:
            resource = stack.enter_context(
                RangeReader(client, item["url"], item["archive_bytes"], budget)
            )
        else:
            raise ValueError(
                "Provide either a local archive or a client and transfer budget."
            )
        archive = stack.enter_context(zipfile.ZipFile(resource))
        names = archive.namelist()
        if len(set(names)) != len(names):
            raise ValueError("Duplicate ZIP member names.")
        patches = [
            i
            for i in archive.infolist()
            if i.filename.endswith("/Fixed.patch") or i.filename == "Fixed.patch"
        ]
        if len(patches) != 1:
            raise ValueError(
                f"Expected one canonical Fixed.patch; found {len(patches)}."
            )
        patch_member = patches[0]
        payload = bounded_member(archive, patch_member)
        patch = payload.decode(
            "utf-8"
        )  # Ambiguous decoding is quarantined, never silently replaced.
        parsed = PatchSet(patch)
        if not parsed or len(parsed) > 10_000:
            raise ValueError("Empty patch or more than 10,000 changed files.")
        root = patch_member.filename.removesuffix("Fixed.patch")
        snapshot = [
            entry
            for entry in archive.infolist()
            if not entry.is_dir() and entry.filename.startswith(root + "Flaky/")
        ]
        snapshot_metrics = {
            "snapshot_files": len(snapshot),
            "snapshot_java_files": sum(
                entry.filename.endswith(".java") for entry in snapshot
            ),
            "snapshot_bytes": sum(entry.file_size for entry in snapshot),
        }
        originals = {}
        for changed in parsed:
            for path in (changed.source_file, changed.target_file):
                patch_path(path)
            if changed.is_added_file or changed.is_binary_file:
                continue
            path = patch_path(changed.source_file)
            member = archive.getinfo(root + "Flaky/" + path)
            if (
                sum(len(s.encode("utf-8")) for s in originals.values())
                + member.file_size
                > 128_000_000
            ):
                raise BudgetExceeded("Original-source text exceeds 128 MB per repair.")
            originals[path] = bounded_member(archive, member).decode("utf-8")
        if not originals:
            raise ValueError("Patch has no original source files.")
        if sum(len(s.encode("utf-8")) for s in originals.values()) > 128_000_000:
            raise BudgetExceeded("Original-source text exceeds 128 MB per repair.")
    result = analyze_patch(patch, originals)
    return {
        "archive_id": item["archive_id"],
        "archive_url": item["url"],
        "publisher_checksum": item["publisher_checksum"],
        "patch_member": patch_member.filename,
        "patch_sha256": sha256(payload),
        "originals_sha256": sha256(
            json.dumps(originals, sort_keys=True).encode("utf-8")
        ),
        "patch": patch,
        "originals": originals,
        **snapshot_metrics,
        **result,
    }


def inspect_corpus(
    cases: pd.DataFrame, catalog: pd.DataFrame, downloads: pd.DataFrame, directory: Path
) -> tuple[pd.DataFrame, list[dict]]:
    directory.mkdir(parents=True, exist_ok=True)
    if set(catalog.archive_id) != set(cases.archive_id):
        raise ValueError("Full catalog/case coverage is required.")
    locations = downloads.set_index("archive_id").local_path
    assignments = cases[["archive_id", "project", "split"]].drop_duplicates()
    catalog = catalog.drop(columns="project", errors="ignore").merge(
        assignments, on="archive_id", validate="one_to_one"
    )
    rows, pairs = [], []
    for item in catalog.to_dict("records"):
        key = item["archive_id"]
        report = {"archive_id": key, "project": item["project"], "split": item["split"]}
        if key not in locations or not Path(locations[key]).is_file():
            raise ValueError(f"Full download is incomplete: {key}")
        try:
            pair = inspect_archive(None, item, None, Path(locations[key]))
            pair.update(project=item["project"], split=item["split"])
            write_json(directory / (sha256(key.encode()) + ".json"), pair)
            pairs.append(pair)
            rows.append(
                {
                    **report,
                    **{
                        k: v for k, v in pair.items() if k not in {"patch", "originals"}
                    },
                    "status": "inspected",
                    "reason": "",
                }
            )
        except (
            ValueError,
            KeyError,
            UnicodeError,
            BudgetExceeded,
            zipfile.BadZipFile,
            UnidiffParseError,
        ) as exc:
            LOGGER.warning("Quarantined archive %s: %s", key, exc)
            rows.append({**report, "status": "quarantined", "reason": str(exc)})
    return pd.DataFrame(rows), pairs


def parse_rounds(payload: bytes, path: str) -> tuple[list[dict], list[dict]]:
    parts = PurePosixPath(path).parts
    if len(parts) < 5 or parts[0] != "result" or parts[2] != "result":
        raise ValueError(f"Unexpected execution-summary path: {path}")
    rows, issues = [], []
    status_map = {
        "success": "passed",
        "pass": "passed",
        "passed": "passed",
        "failure": "failed",
        "fail": "failed",
        "failed": "failed",
        "error": "error",
        "skipped": "skipped",
        "skip": "skipped",
    }
    previous_limit = csv.field_size_limit(64_000_000)
    try:
        source_rows = list(csv.reader(io.StringIO(payload.decode("utf-8-sig"))))
    finally:
        csv.field_size_limit(previous_limit)
    version = {
        "FixedPssingOrder": "FixedPassingOrder",
        "FlakyPssingOrder": "FlakyPassingOrder",
    }.get(parts[3], parts[3])
    for line, row in enumerate(source_rows, 1):
        if not row:
            continue
        if row[0].strip().lower() in {"test", "test_name", "testname", "iteration"}:
            continue
        measurements = []
        duration_format_normalized = False
        if len(row) == 5:
            test, status, seconds, iteration, report = (value.strip() for value in row)
            measurements.append(
                (
                    test,
                    status,
                    seconds,
                    iteration,
                    report,
                    "0",
                    "reported_index",
                    "test",
                )
            )
        elif len(row) == 4:
            try:
                details = json.loads(row[3])
                if not isinstance(details, dict) or not isinstance(
                    details.get("results"), dict
                ):
                    raise ValueError("Missing embedded invocation results")
                if not isinstance(details.get("id"), str):
                    raise ValueError("Missing embedded execution identity")
                for invocation, result in details["results"].items():
                    if not isinstance(result, dict) or not {
                        "name",
                        "result",
                        "time",
                    } <= set(result):
                        raise ValueError("Incomplete embedded invocation")
                    measurements.append(
                        (
                            result["name"],
                            result["result"],
                            result["time"],
                            line - 1,
                            details["id"],
                            invocation,
                            "summary_row_index",
                            "test",
                        )
                    )
            except (ValueError, TypeError) as exc:
                issues.append(
                    {
                        "path": path,
                        "row": line,
                        "issue": f"Invalid embedded execution JSON: {exc}",
                    }
                )
                continue
        elif len(row) == 6 and row[1].strip().lower() in {*status_map, "unknown"}:
            duration = ",".join(row[2:4]).strip()
            if not re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", duration):
                issues.append(
                    {
                        "path": path,
                        "row": line,
                        "issue": "Unrecognized split test-duration format",
                    }
                )
                continue
            test, status = (value.strip() for value in row[:2])
            iteration, report = (value.strip() for value in row[4:])
            duration_format_normalized = True
            measurements.append(
                (
                    test,
                    status,
                    duration.replace(",", ""),
                    iteration,
                    report,
                    "0",
                    "reported_index",
                    "test",
                )
            )
        elif len(row) in {6, 7}:
            iteration, execution_id, seed, report, status = (
                value.strip() for value in row[:5]
            )
            duration = ",".join(row[5:]).strip()
            if len(row) == 7 and not re.fullmatch(
                r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", duration
            ):
                issues.append(
                    {
                        "path": path,
                        "row": line,
                        "issue": "Unrecognized split duration format",
                    }
                )
                continue
            measurements.append(
                (
                    None,
                    status,
                    duration.replace(",", ""),
                    iteration,
                    report,
                    execution_id,
                    "reported_index",
                    "report",
                )
            )
            duration_format_normalized = len(row) == 7
        else:
            issues.append(
                {
                    "path": path,
                    "row": line,
                    "issue": f"Unrecognized summary schema: {len(row)} fields",
                }
            )
            continue
        for (
            test,
            status,
            seconds,
            iteration,
            report,
            invocation,
            round_source,
            identity_scope,
        ) in measurements:
            try:
                round_value = int(iteration)
                if round_value < 0:
                    raise ValueError("Negative iteration")
                if (
                    identity_scope == "test" and not isinstance(test, str)
                ) or not isinstance(status, str):
                    raise ValueError("Non-text test identity or outcome")
            except (ValueError, TypeError) as exc:
                issues.append(
                    {
                        "path": path,
                        "row": line,
                        "issue": f"Invalid execution measurement: {exc}",
                    }
                )
                continue
            try:
                seconds_value = float(seconds)
                if not np.isfinite(seconds_value) or seconds_value < 0:
                    raise ValueError("Negative/nonfinite duration")
            except (ValueError, TypeError) as exc:
                seconds_value = None
                issues.append(
                    {
                        "path": path,
                        "row": line,
                        "issue": f"Duration set missing; outcome retained: {exc}",
                    }
                )
            normalized = status_map.get(status.lower(), "unknown")
            if normalized == "unknown":
                issues.append(
                    {"path": path, "row": line, "issue": f"Unknown outcome: {status}"}
                )
            rows.append(
                {
                    "case_id": parts[1],
                    "version": version,
                    "version_raw": parts[3],
                    "test_name": test.replace("#", ".") if test is not None else None,
                    "reported_class": (
                        PurePosixPath(report).name[5:-4]
                        if identity_scope == "report"
                        and PurePosixPath(report).name.startswith("TEST-")
                        and report.endswith(".xml")
                        else None
                    ),
                    "identity_scope": identity_scope,
                    "invocation": invocation,
                    "execution_seed": (
                        row[2].strip() if identity_scope == "report" else None
                    ),
                    "outcome": normalized,
                    "seconds": seconds_value,
                    "round": round_value,
                    "round_source": round_source,
                    "report_path": report,
                    "summary_path": path,
                    "duration_format_normalized": duration_format_normalized,
                }
            )
    return rows, issues


def execution_results(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read the whole ZIP sequentially, avoiding its 356 MB Python ZIP index."""
    inventory, rounds, issues = {}, [], []
    summaries_seen = set()
    with libarchive.file_reader(str(path), format_name="zip") as archive:
        for entry in archive:
            name = entry.pathname
            parts = PurePosixPath(name).parts
            if not entry.isfile:
                continue
            case = (
                parts[1] if len(parts) > 1 and parts[0] == "result" else "__unmapped__"
            )
            version = (
                parts[3] if len(parts) > 3 and parts[2] == "result" else "__metadata__"
            )
            key = (case, version)
            item = inventory.setdefault(
                key,
                {
                    "case_id": case,
                    "version": version,
                    "files": 0,
                    "bytes": 0,
                    "log_files": 0,
                    "log_bytes": 0,
                    "log_lines": 0,
                    "log_replacement_characters": 0,
                    "round_summary_files": 0,
                },
            )
            item["files"] += 1
            item["bytes"] += entry.size or 0
            if name.endswith(".log"):
                item["log_files"] += 1
                item["log_bytes"] += entry.size or 0
                decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
                lines, read_bytes, final_byte, replacements = 0, 0, b"", 0
                for block in entry.get_blocks():
                    read_bytes += len(block)
                    lines += block.count(b"\n")
                    final_byte = block[-1:]
                    replacements += decoder.decode(block).count("\ufffd")
                replacements += decoder.decode(b"", final=True).count("\ufffd")
                if read_bytes != entry.size:
                    raise ValueError(f"Truncated log payload: {name}")
                item["log_lines"] += lines + int(
                    bool(final_byte) and final_byte != b"\n"
                )
                item["log_replacement_characters"] += replacements
                continue
            if not name.endswith("/rounds-test-results.csv"):
                continue
            item["round_summary_files"] += 1
            try:
                safe_path(name)
                if name in summaries_seen:
                    raise ValueError("Duplicate execution-summary archive path")
                summaries_seen.add(name)
                if entry.size > 64_000_000:
                    raise ValueError("Execution-summary member exceeds 64 MB")
                payload = bytearray()
                for block in entry.get_blocks():
                    if len(payload) + len(block) > 64_000_000:
                        raise ValueError("Execution-summary payload exceeds 64 MB")
                    payload.extend(block)
                parsed, invalid = parse_rounds(bytes(payload), name)
                rounds.extend(parsed)
                issues.extend(invalid)
            except (ValueError, UnicodeError, csv.Error) as exc:
                issues.append({"path": name, "row": None, "issue": str(exc)})
    if not inventory or not rounds:
        raise ValueError(
            "Execution corpus contains no file inventory or parsed trial evidence."
        )
    return (
        pd.DataFrame(inventory.values()),
        pd.DataFrame(rounds),
        pd.DataFrame(issues, columns=["path", "row", "issue"]),
    )


def analyze_patch(patch: str, originals: dict[str, str]) -> dict:
    parsed = PatchSet(patch)
    targets = [patch_path(f.target_file) for f in parsed if not f.is_removed_file]
    removed = [
        line.value for f in parsed for hunk in f for line in hunk if line.is_removed
    ]
    added = [line.value for f in parsed for hunk in f for line in hunk if line.is_added]
    check = None
    with tempfile.TemporaryDirectory(prefix="reproflake-patch-") as directory:
        root = Path(directory)
        for name, content in originals.items():
            path = root / safe_path(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content.encode("utf-8"))
        patch_file = root / "reference.patch"
        patch_file.write_text(patch, encoding="utf-8")
        checked = subprocess.run(
            ["git", "apply", "--check", "-p1", str(patch_file)],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        check = checked.returncode == 0
        check_reason = checked.stderr.strip()
    if checked.returncode not in {0, 1, 128}:
        raise RuntimeError(
            f"Unexpected git apply exit {checked.returncode}: {check_reason}"
        )
    assertion = re.compile(r"\b(?:assert\w*|fail)\s*\(")
    paths = [
        patch_path(f.source_file if f.is_removed_file else f.target_file)
        for f in parsed
    ]
    binary_count = sum(f.is_binary_file for f in parsed)
    if binary_count:
        check = False
        check_reason = "Reference contains non-reconstructible binary differences; not a text-only repair."
    return {
        "files_changed": len(parsed),
        "hunks": sum(len(f) for f in parsed),
        "binary_files_changed": binary_count,
        "generated_files_changed": sum(
            "/target/" in "/" + p or "/build/" in "/" + p for p in paths
        ),
        "lines_added": len(added),
        "lines_removed": len(removed),
        "changed_lines": len(added) + len(removed),
        "patch_characters": len(patch),
        "source_lines": sum(len(s.splitlines()) for s in originals.values()),
        "source_characters": sum(len(s) for s in originals.values()),
        "touches_test": any("/src/test/" in "/" + p for p in paths),
        "touches_production": any("/src/main/" in "/" + p for p in paths),
        "removed_assertion_lines": sum(
            bool(assertion.search(line)) for line in removed
        ),
        "added_assertion_lines": sum(bool(assertion.search(line)) for line in added),
        "adds_skip_marker": any(
            re.search(r"@(?:Ignore|Disabled)\b", line) for line in added
        ),
        "adds_sleep": any("Thread.sleep(" in line for line in added),
        "deletes_test_file": any(
            f.is_removed_file and "/src/test/" in "/" + patch_path(f.source_file)
            for f in parsed
        ),
        "applies_to_snapshot": check,
        "apply_check_reason": check_reason,
        "changed_paths": paths,
        "target_paths": targets,
    }


def inspect_sample(
    client: requests.Session,
    selected: pd.DataFrame,
    catalog: pd.DataFrame,
    directory: Path,
    transfer_limit: int,
) -> tuple[pd.DataFrame, list[dict], int]:
    directory.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(directory).free < transfer_limit + 256_000_000:
        raise OSError("Insufficient free disk space for configured inspection budget.")
    budget = TransferBudget(transfer_limit)
    rows, pairs = [], []
    catalog = catalog.set_index("archive_id")
    stopped = False
    for selected_row in selected.to_dict("records"):
        key = selected_row["archive_id"]
        item = catalog.loc[key].to_dict()
        item["archive_id"] = key
        cache = directory / (sha256(key.encode()) + ".json")
        report = {
            "archive_id": key,
            "project": selected_row["project"],
            "split": selected_row["split"],
        }
        if stopped:
            rows.append(
                {
                    **report,
                    "status": "not_attempted",
                    "reason": "Transfer cap exhausted",
                }
            )
            continue
        try:
            if cache.exists():
                pair = json.loads(cache.read_text())
                if (
                    pair["archive_url"] != item["url"]
                    or pair["publisher_checksum"] != item["publisher_checksum"]
                    or pair["patch_sha256"] != sha256(pair["patch"].encode("utf-8"))
                    or pair["originals_sha256"]
                    != sha256(
                        json.dumps(pair["originals"], sort_keys=True).encode("utf-8")
                    )
                ):
                    raise ValueError("Modified or stale repair-pair cache.")
                pair.update(analyze_patch(pair["patch"], pair["originals"]))
            else:
                pair = inspect_archive(client, item, budget)
                write_json(cache, pair)
            pair.update(project=report["project"], split=report["split"])
            pairs.append(pair)
            rows.append(
                {
                    **report,
                    **{
                        k: v for k, v in pair.items() if k not in {"patch", "originals"}
                    },
                    "status": "inspected",
                    "reason": "",
                }
            )
        except BudgetExceeded as exc:
            LOGGER.warning("%s: %s", key, exc)
            stopped = budget.used >= budget.limit or "transfer cap" in str(exc).lower()
            rows.append({**report, "status": "limited", "reason": str(exc)})
        except requests.RequestException as exc:
            LOGGER.warning("Retrieval failed for %s: %s", key, exc)
            rows.append({**report, "status": "retrieval_failed", "reason": str(exc)})
        except (
            ValueError,
            KeyError,
            UnicodeError,
            zipfile.BadZipFile,
            UnidiffParseError,
        ) as exc:
            LOGGER.warning("Quarantined archive %s: %s", key, exc)
            rows.append({**report, "status": "quarantined", "reason": str(exc)})
    return pd.DataFrame(rows), pairs, budget.used


def export_pairs(pairs: list[dict], directory: Path) -> pd.DataFrame:
    """Keep solution-bearing evidence separate from blind evaluation case manifests."""
    directory.mkdir(parents=True, exist_ok=True)
    audit = []
    seen = set()
    eligible = []
    patch_splits = {}
    for pair in pairs:
        patch_splits.setdefault(pair["patch_sha256"], set()).add(pair["split"])
    for pair in pairs:
        signature = (pair["originals_sha256"], pair["patch_sha256"])
        reason = "eligible"
        if len(patch_splits[pair["patch_sha256"]]) > 1:
            reason = "quarantined: shared reference patch crosses partitions"
        elif signature in seen:
            reason = "duplicate source/patch pair"
        elif not pair["applies_to_snapshot"]:
            reason = "quarantined: reference patch does not apply"
        elif pair.get("generated_files_changed", 0):
            reason = "quarantined: generated/build-artifact edits require review"
        elif pair["adds_skip_marker"] or pair["deletes_test_file"]:
            reason = "quarantined: test suppression requires review"
        seen.add(signature)
        audit.append(
            {
                "archive_id": pair["archive_id"],
                "project": pair["project"],
                "split": pair["split"],
                "decision": reason,
            }
        )
        if reason == "eligible":
            eligible.append(pair)
    for split in ("train", "validation", "test"):
        temporary = directory / f"{split}_reference_pairs.jsonl.part"
        with temporary.open("w", encoding="utf-8") as stream:
            for pair in eligible:
                if pair["split"] == split:
                    record = {
                        "archive_id": pair["archive_id"],
                        "project": pair["project"],
                        "original_files": pair["originals"],
                        "target_patch": pair["patch"],
                        "patch_sha256": pair["patch_sha256"],
                        "context_policy": "Patch-guided changed originals: training/reference evidence ONLY; never use as blind evaluation context.",
                        "validation": "Static git apply check only; not a compiled or behaviorally verified repair.",
                    }
                    stream.write(json.dumps(record) + "\n")
        temporary.replace(directory / f"{split}_reference_pairs.jsonl")
    return pd.DataFrame(audit)


def correlations(
    frame: pd.DataFrame, columns: list[str], minimum: int = 15
) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric = frame[columns].astype(float)
    counts = numeric.notna().astype(int).T @ numeric.notna().astype(int)
    corr = numeric.corr(method="spearman", min_periods=minimum)
    rows = []
    for i, left in enumerate(columns):
        for right in columns[i + 1 :]:
            coefficient = corr.loc[left, right]
            if pd.notna(coefficient):
                rows.append(
                    {
                        "left": left,
                        "right": right,
                        "spearman": coefficient,
                        "paired_repairs": int(counts.loc[left, right]),
                        "projects": frame.loc[
                            numeric[left].notna() & numeric[right].notna(), "project"
                        ].nunique(),
                    }
                )
    return corr, pd.DataFrame(
        rows, columns=["left", "right", "spearman", "paired_repairs", "projects"]
    )


def project_bootstrap(
    frame: pd.DataFrame, left: str, right: str, seed: int, repeats: int = 500
) -> tuple[float, float]:
    groups = [g for _, g in frame.dropna(subset=[left, right]).groupby("project")]
    if len(groups) < 10:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        sampled = pd.concat(
            [groups[i] for i in rng.integers(0, len(groups), len(groups))],
            ignore_index=True,
        )
        if sampled[left].nunique() > 1 and sampled[right].nunique() > 1:
            values.append(sampled[[left, right]].corr(method="spearman").iloc[0, 1])
    if len(values) < repeats * 0.8:
        raise ValueError("Too many degenerate project-bootstrap resamples.")
    return tuple(float(x) for x in np.quantile(values, [0.025, 0.975]))


def paired_execution_summary(
    executions: pd.DataFrame,
    cases: pd.DataFrame,
    seed: int = 590,
    repeats: int = 500,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare the same cases and evidence scope without pooling suite/test rates."""
    if repeats < 2:
        raise ValueError("At least two project-bootstrap repetitions are required.")
    keys = ["case_id", "identity_scope"]
    base = executions.loc[executions.version.isin(["Flaky", "Fixed"])].copy()
    if not base.identity_scope.isin(["test", "report"]).all():
        raise ValueError("Use only named-test or report-level execution evidence.")
    if base.duplicated(keys + ["version"]).any():
        raise ValueError(
            "Execution summaries must be unique per case, condition, and scope."
        )
    if (
        base.recorded_trials.le(0).any()
        or not base.failures.between(0, base.recorded_trials).all()
    ):
        raise ValueError("Execution counts must be positive with valid failure totals.")
    metrics = ["recorded_trials", "failures"]
    original = base.loc[base.version.eq("Flaky"), keys + metrics].rename(
        columns={"recorded_trials": "original_trials", "failures": "original_failures"}
    )
    fixed = base.loc[base.version.eq("Fixed"), keys + metrics].rename(
        columns={"recorded_trials": "fixed_trials", "failures": "fixed_failures"}
    )
    paired = original.merge(fixed, on=keys, validate="one_to_one")
    if cases.case_id.duplicated().any() or not set(paired.case_id) <= set(
        cases.case_id
    ):
        raise ValueError("Paired cases require unique, complete project provenance.")
    paired = paired.merge(
        cases[["case_id", "project", "category"]], on="case_id", validate="many_to_one"
    )
    if paired.empty:
        raise ValueError("No cases have both original and fixed execution summaries.")
    paired["original_rate"] = paired.original_failures / paired.original_trials
    paired["fixed_rate"] = paired.fixed_failures / paired.fixed_trials
    paired["rate_reduction"] = paired.original_rate - paired.fixed_rate
    summaries = []
    for scope, group in paired.groupby("identity_scope", sort=True):
        clusters = group.groupby("project").rate_reduction.agg(["sum", "count"])
        if len(clusters) >= 10:
            rng = np.random.default_rng(seed)
            draw = rng.integers(0, len(clusters), size=(repeats, len(clusters)))
            reductions = clusters["sum"].to_numpy()[draw].sum(axis=1) / clusters[
                "count"
            ].to_numpy()[draw].sum(axis=1)
            low, high = np.quantile(reductions, [0.025, 0.975])
            interval_status = "project-composition stability"
        else:
            low, high = np.nan, np.nan
            interval_status = "not estimated: fewer than 10 projects"
        summaries.append(
            {
                "identity_scope": scope,
                "original_cases": original.loc[
                    original.identity_scope.eq(scope), "case_id"
                ].nunique(),
                "fixed_cases": fixed.loc[
                    fixed.identity_scope.eq(scope), "case_id"
                ].nunique(),
                "paired_cases": len(group),
                "projects": group.project.nunique(),
                "original_case_mean": group.original_rate.mean(),
                "fixed_case_mean": group.fixed_rate.mean(),
                "original_pooled_rate": group.original_failures.sum()
                / group.original_trials.sum(),
                "fixed_pooled_rate": group.fixed_failures.sum()
                / group.fixed_trials.sum(),
                "mean_reduction": group.rate_reduction.mean(),
                "reduction_low": low,
                "reduction_high": high,
                "interval_status": interval_status,
                "fixed_cases_with_failures": int(group.fixed_failures.gt(0).sum()),
                "fixed_cases_without_failures": int(group.fixed_failures.eq(0).sum()),
            }
        )
    return paired, pd.DataFrame(summaries)


def correlation_sensitivity(
    repairs: pd.DataFrame,
    audit: pd.DataFrame,
    columns: list[str],
    minimum: int = 15,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if repairs.archive_id.duplicated().any() or audit.archive_id.duplicated().any():
        raise ValueError("Sensitivity inputs must have unique archive identities.")
    if not set(repairs.archive_id) <= set(audit.archive_id):
        raise ValueError("Every analyzed repair requires an export-audit decision.")
    cutoff = repairs.changed_lines.quantile(0.99)
    eligible = set(audit.loc[audit.decision.eq("eligible"), "archive_id"])
    cohorts = {
        "Full corpus": repairs,
        "Mechanically screened": repairs.loc[repairs.archive_id.isin(eligible)],
        "Below 99th-percentile edit size": repairs.loc[
            repairs.changed_lines.le(cutoff)
        ],
    }
    rows, coverage = [], []
    for name, frame in cohorts.items():
        coverage.append(
            {
                "cohort": name,
                "repairs": len(frame),
                "projects": frame.project.nunique(),
                "excluded_repairs": len(repairs) - len(frame),
                "edit_size_cutoff": cutoff if name.startswith("Below") else np.nan,
            }
        )
        _, associations = correlations(frame, columns, minimum)
        associations["cohort"] = name
        rows.append(associations)
    return pd.concat(rows, ignore_index=True), pd.DataFrame(coverage)
