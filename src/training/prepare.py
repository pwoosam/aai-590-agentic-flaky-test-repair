"""Read original Java context from local ReproFlake ZIPs; never execute it."""

import argparse
import logging
import re
import stat
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from zipfile import ZipFile, ZipInfo

from .data import (
    BUNDLE_VERSION,
    SPLITS,
    Assignment,
    digest,
    file_digest,
    index,
    read_csv,
    validate_assignments,
    validate_repair_splits,
    write_json,
    write_jsonl,
)

LOG = logging.getLogger(__name__)
IDENTIFIER = r"[A-Za-z_$][\w$]*"
QUALIFIED = rf"{IDENTIFIER}(?:\.{IDENTIFIER})*"
TEST = re.compile(rf"({QUALIFIED})#({IDENTIFIER})(?:\[[^\r\n\]]*\])?")
IMPORT = re.compile(rf"(?m)^\s*import\s+({QUALIFIED})\s*;")
SOURCE_BYTES = 512_000
PATCH_BYTES = 16_000_000
SOURCE_ERRORS = (KeyError, ValueError)


def safe_relative(value: str) -> bool:
    path = PurePosixPath(value)
    return (
        bool(value)
        and not path.is_absolute()
        and not {"..", ".git"}.intersection(path.parts)
        and not re.search(r"[\x00-\x1f\\:]", value)
    )


def member_bytes(archive: ZipFile, entry: ZipInfo, limit: int) -> bytes:
    if (
        not safe_relative(entry.filename)
        or stat.S_ISLNK(entry.external_attr >> 16)
        or max(entry.file_size, entry.compress_size) > limit
    ):
        raise ValueError(f"Unsafe or oversized member: {entry.filename}")
    return archive.read(entry)


class OriginalSnapshot:
    def __init__(self, archive: ZipFile, archive_id: str):
        self.archive = archive
        self.prefix = f"{archive_id}/Flaky/"
        self.by_name: dict[str, list[str]] = defaultdict(list)
        self.files = {}
        for entry in archive.infolist():
            if entry.is_dir() or not entry.filename.startswith(self.prefix):
                continue
            path = entry.filename[len(self.prefix) :]
            if not safe_relative(path):
                continue
            if path in self.files:
                raise ValueError(f"Duplicate original ZIP member: {path}")
            self.files[path] = entry
            if path.endswith((".java", ".groovy")):
                self.by_name[PurePosixPath(path).stem].append(path)
        if not self.files:
            raise ValueError(f"No original Flaky/ snapshot in {archive_id}")
        self.text: dict[str, str] = {}

    def read(self, path: str) -> str:
        if path not in self.text:
            payload = member_bytes(self.archive, self.files[path], SOURCE_BYTES)
            self.text[path] = payload.decode("utf-8")
        return self.text[path]

    def jvm_path(self, qualified: str, module: str, method: str = "") -> str:
        suffix = qualified.split("$", 1)[0].replace(".", "/")
        candidates = [
            path
            for path in self.by_name[PurePosixPath(suffix).name]
            if (stem := path.rsplit(".", 1)[0]) == suffix or stem.endswith("/" + suffix)
        ]
        if not candidates:
            raise ValueError(f"No original Java/Groovy file for {qualified}")

        def rank(path: str) -> tuple[bool, bool, bool]:
            return (
                module != "." and not path.startswith(module + "/"),
                "/src/test/" not in f"/{path}",
                "/src/" not in f"/{path}",
            )

        best = min(map(rank, candidates))
        matches = [path for path in candidates if rank(path) == best]
        if len(matches) > 1 and method:
            pattern = re.compile(rf"\b{re.escape(method)}\s*\(")
            anchored = [path for path in matches if pattern.search(self.read(path))]
            if len(anchored) == 1:
                return anchored[0]
        if len(matches) != 1:
            raise ValueError(f"Ambiguous original Java/Groovy file for {qualified}")
        return matches[0]

    def context(self, row: dict[str, str]) -> dict:
        test = TEST.fullmatch(row["test_name"])
        if test is None:
            raise ValueError("Unsupported test identifier")
        qualified, method = test.groups()
        module = row["module"] or "."
        if not safe_relative(module):
            raise ValueError("Unsafe module path")
        path = self.jvm_path(qualified, module, method)
        source = self.read(path)
        if not source.strip():
            raise ValueError("Empty failing test source")
        files = [
            {"path": path, "text": source, "kind": "failing test", "method": method}
        ]
        seen = {path}

        def include(candidate: str, kind: str, anchor: str = "") -> bool:
            if candidate in seen or candidate not in self.files:
                return False
            try:
                text = self.read(candidate)
            except SOURCE_ERRORS as exc:
                LOG.warning("Omitting optional %s %s: %s", kind, candidate, exc)
                return False
            files.append(
                {"path": candidate, "text": text, "kind": kind, "method": anchor}
            )
            seen.add(candidate)
            return True

        polluter = TEST.fullmatch(row.get("polluter", ""))
        if polluter:
            try:
                candidate = self.jvm_path(polluter[1], module, polluter[2])
            except ValueError:
                candidate = ""
            include(candidate, "reported polluter", polluter[2])
        pom = str(PurePosixPath(module) / "pom.xml")
        if not include(pom, "Maven configuration"):
            include("pom.xml", "Maven configuration")
        imported = 0
        for name in dict.fromkeys(IMPORT.findall(source)):
            try:
                candidate = self.jvm_path(name, module)
            except ValueError:
                continue
            if include(candidate, "repository import"):
                imported += 1
                if imported == 3:
                    break
        environment = {}
        if row["module"]:
            environment["Maven module"] = module
        for field, label in (
            ("java_version", "Java version"),
            ("iterations", "Configured repetitions"),
            ("nondex_seed", "NonDex seed"),
        ):
            value = row.get(field, "")
            if re.fullmatch(r"[0-9]{1,20}", value):
                environment[label] = value
        if polluter:
            environment["Reported polluter"] = f"{polluter[1]}#{polluter[2]}"
        return {
            "test": f"{qualified}#{method}",
            "environment": environment,
            "files": files,
        }


def original_outcomes(eda: Path) -> dict:
    path = eda / "case_execution_summary.csv"
    if not path.is_file():
        LOG.info(
            "No case_execution_summary.csv; original failure counts will be omitted."
        )
        return {}
    rows = index(
        (
            row
            for row in read_csv(path)
            if row["version"] == "Flaky" and row["identity_scope"] == "test"
        ),
        "case_id",
    )
    results = {}
    for case, row in rows.items():
        failures, trials = int(row["failures"]), int(row["recorded_trials"])
        if not 0 <= failures <= trials or not trials:
            raise ValueError(f"Invalid recorded Flaky trials for {case}")
        results[case] = f"{failures} failures / {trials} recorded test trials"
    return results


def prepare(archives: Path, eda: Path, output: Path) -> dict:
    assignment_path = eda / "split_assignments.csv"
    cases = validate_assignments(read_csv(assignment_path))
    inventory = index(read_csv(eda / "case_inventory.csv"), "case_id")
    if cases != {key: Assignment.from_row(row) for key, row in inventory.items()}:
        raise ValueError("Case inventory does not match canonical split assignments.")
    catalog = index(read_csv(eda / "archive_catalog.csv"), "archive_id")
    audit = index(read_csv(eda / "pair_export_audit.csv"), "archive_id")
    inspection = index(read_csv(eda / "archive_inspection.csv"), "archive_id")
    by_archive = defaultdict(list)
    for case in cases.values():
        by_archive[case.archive_id].append(case)
    if set(by_archive) != set(catalog) or set(by_archive) != set(inspection):
        raise ValueError("Catalog/inspection must cover the entire split manifest.")
    for archive_id, archive_cases in by_archive.items():
        case = archive_cases[0]
        for table in (inspection, audit):
            row = table.get(archive_id)
            if row and (row["project"], row["split"]) != (case.project, case.split):
                raise ValueError(f"Archive partition mismatch: {archive_id}")
    eligible = {
        archive_id for archive_id, row in audit.items() if row["decision"] == "eligible"
    }
    if not set(audit) <= set(by_archive):
        raise ValueError("Pair audit contains unassigned archives.")
    inspected_hashes = {
        archive_id: row["patch_sha256"]
        for archive_id, row in inspection.items()
        if row["status"] == "inspected"
    }
    if not eligible <= inspected_hashes.keys():
        raise ValueError("An eligible archive lacks a verified reference patch.")
    shared = defaultdict(set)
    for archive_id, checksum in inspected_hashes.items():
        shared[checksum].add(by_archive[archive_id][0].split)
    if any(len(shared[inspected_hashes[archive_id]]) != 1 for archive_id in eligible):
        raise ValueError("A cross-partition reference was not quarantined by the EDA.")
    outcomes = original_outcomes(eda)
    inputs = {split: [] for split in SPLITS}
    targets = {split: [] for split in SPLITS}
    exclusions = []
    target_hashes = {}
    for archive_id, archive_cases in sorted(by_archive.items()):
        split = archive_cases[0].split
        if archive_id not in eligible:
            decision = (
                audit.get(archive_id, {}).get("decision") or "Not inspected/eligible"
            )
            exclusions.extend(
                {**vars(case), "reason": decision} for case in archive_cases
            )
            continue
        checksum = inspected_hashes[archive_id]
        filename = catalog[archive_id]["file_name"]
        if (
            filename != f"{archive_id}.zip"
            or not safe_relative(filename)
            or "/" in filename
        ):
            raise ValueError(f"Unexpected archive filename: {filename}")
        path = archives / filename
        if path.stat().st_size != int(catalog[archive_id]["archive_bytes"]):
            raise ValueError(f"Archive size differs from acquisition catalog: {path}")
        archive_inputs = []
        with ZipFile(path) as archive:
            snapshot = OriginalSnapshot(archive, archive_id)
            for case in sorted(archive_cases, key=lambda case: case.case_id):
                try:
                    context = snapshot.context(inventory[case.case_id])
                except (ValueError, KeyError) as exc:
                    exclusions.append({**vars(case), "reason": str(exc)})
                    continue
                if case.case_id in outcomes:
                    context["environment"]["Published original outcomes"] = outcomes[
                        case.case_id
                    ]
                archive_inputs.append({**vars(case), **context})
            if archive_inputs:
                patch_name = f"{archive_id}/Fixed.patch"
                if archive.namelist().count(patch_name) != 1:
                    raise ValueError(f"Expected one canonical reference in {path}")
                payload = member_bytes(
                    archive, archive.getinfo(patch_name), PATCH_BYTES
                )
                if digest(payload) != checksum:
                    raise ValueError(
                        f"Canonical reference hash differs from inspection: {path}"
                    )
                patch = payload.decode("utf-8")
                inputs[split].extend(archive_inputs)
                targets[split].append({"archive_id": archive_id, "patch": patch})
                target_hashes[archive_id] = checksum
    validate_repair_splits(target_hashes, cases)
    if any(not inputs[split] for split in SPLITS):
        raise ValueError("Prepared data must have usable inputs in every split.")
    output.mkdir(parents=True, exist_ok=False)
    write_jsonl(output / "assignments.jsonl", (vars(case) for case in cases.values()))
    for split in SPLITS:
        write_jsonl(output / f"{split}_inputs.jsonl", inputs[split])
        write_jsonl(output / f"{split}_targets.jsonl", targets[split])
    write_jsonl(output / "exclusions.jsonl", exclusions)
    summary = {
        split: {
            "assigned_cases": sum(case.split == split for case in cases.values()),
            "prepared_cases": len(inputs[split]),
            "prepared_archives": len(targets[split]),
        }
        for split in SPLITS
    }
    write_json(
        output / "manifest.json",
        {
            "version": BUNDLE_VERSION,
            "split_csv_sha256": file_digest(assignment_path),
            "summary": summary,
            "target_hashes": target_hashes,
            "files": {
                path.name: file_digest(path) for path in sorted(output.glob("*.jsonl"))
            },
        },
    )
    LOG.info("Prepared split coverage: %s", summary)
    LOG.info("Exclusions by reason: %s", Counter(item["reason"] for item in exclusions))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archives", type=Path, default=Path("data/reproflake/archives")
    )
    parser.add_argument("--eda", type=Path, default=Path("outputs/reproflake"))
    parser.add_argument(
        "--output", type=Path, required=True, help="New prepared bundle directory"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    prepare(args.archives, args.eda, args.output)


if __name__ == "__main__":
    main()
