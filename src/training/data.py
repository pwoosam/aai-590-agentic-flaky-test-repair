"""Split validation and the prepared ReproFlake bundle contract."""

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

SPLITS = ("train", "validation", "test")
EXPECTED_COUNTS = {
    "train": (98, 869, 222),
    "validation": (21, 64, 32),
    "test": (21, 182, 46),
}
BUNDLE_VERSION = 1


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def read_json(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def index(rows, key: str) -> dict:
    result = {}
    for row in rows:
        value = row[key]
        if not value or value in result:
            raise ValueError(f"Missing or duplicate {key}: {value!r}")
        result[value] = row
    return result


@dataclass(frozen=True)
class Assignment:
    case_id: str
    project: str
    archive_id: str
    split: str

    @classmethod
    def from_row(cls, row: dict):
        return cls(**{name: row[name] for name in cls.__dataclass_fields__})


def validate_assignments(rows, *, require_full: bool = True):
    cases = {}
    projects = {}
    archives = {}
    for row in rows:
        case = Assignment.from_row(row)
        if any(
            not isinstance(value, str) or not value.strip()
            for value in vars(case).values()
        ):
            raise ValueError(f"Invalid case assignment: {case}")
        if case.case_id in cases or case.split not in SPLITS:
            raise ValueError(f"Duplicate case or unknown split: {case}")
        if projects.setdefault(case.project.casefold(), case.split) != case.split:
            raise ValueError(f"Project crosses partitions: {case.project}")
        archive_owner = (case.project, case.split)
        if archives.setdefault(case.archive_id, archive_owner) != archive_owner:
            raise ValueError(f"Inconsistent archive ownership: {case.archive_id}")
        cases[case.case_id] = case
    for split in SPLITS:
        subset = [case for case in cases.values() if case.split == split]
        counts = (
            len({case.project.casefold() for case in subset}),
            len(subset),
            len({case.archive_id for case in subset}),
        )
        if not subset or (require_full and counts != EXPECTED_COUNTS[split]):
            raise ValueError(
                f"Unexpected {split} split (projects, cases, archives): {counts}"
            )
    return cases


def validate_repair_splits(target_hashes: dict[str, str], cases: dict[str, Assignment]):
    splits = {case.archive_id: case.split for case in cases.values()}
    seen = {}
    for archive, checksum in target_hashes.items():
        if archive not in splits:
            raise ValueError(f"Target archive is not assigned: {archive}")
        if seen.setdefault(checksum, splits[archive]) != splits[archive]:
            raise ValueError(f"Identical reference patch crosses partitions: {archive}")


class Bundle:
    def __init__(self, path: Path):
        self.path = path
        self.manifest = read_json(path / "manifest.json")
        if self.manifest["version"] != BUNDLE_VERSION:
            raise ValueError(
                "Unsupported prepared bundle version. Re-run data preparation."
            )
        self.fingerprint = file_digest(path / "manifest.json")
        self.cases = validate_assignments(self._read("assignments.jsonl"))
        validate_repair_splits(self.manifest["target_hashes"], self.cases)

    def _read(self, name: str):
        path = self.path / name
        if file_digest(path) != self.manifest["files"][name]:
            raise ValueError(f"Prepared data checksum mismatch: {path}")
        return read_jsonl(path)

    def inputs(self, split: str) -> list[dict]:
        if split not in SPLITS:
            raise ValueError(f"Unknown split: {split}")
        rows = self._read(f"{split}_inputs.jsonl")
        for row in index(rows, "case_id").values():
            case = Assignment.from_row(row)
            if (
                self.cases.get(case.case_id) != case
                or case.split != split
                or case.archive_id not in self.manifest["target_hashes"]
            ):
                raise ValueError(f"Input is inconsistent with prepared split: {case}")
        return rows

    def targets(self, split: str, inputs: list[dict]) -> dict[str, str]:
        if split not in SPLITS:
            raise ValueError(f"Unknown split: {split}")
        expected = {row["archive_id"] for row in inputs}
        rows = index(self._read(f"{split}_targets.jsonl"), "archive_id")
        if set(rows) != expected:
            raise ValueError(f"Input/target archive coverage differs in {split}")
        targets = {}
        for archive, row in rows.items():
            if (
                digest(row["patch"].encode("utf-8"))
                != self.manifest["target_hashes"][archive]
            ):
                raise ValueError(f"Target hash mismatch: {archive}")
            targets[archive] = row["patch"]
        return targets
