# aai-590-agentic-flaky-test-repair

**AAI 590 Capstone: Multi-Agent Flaky-Test Repair for CI/CD**

This repository supports a capstone paper on reproducing and repairing flaky tests
in Java/Maven CI pipelines. The proposed system uses the Microsoft Agent Framework
to coordinate failure reproduction, patch proposal, and isolated validation.
Qwen2.5-Coder-7B-Instruct will be fine-tuned on existing source code and fixing
patches from ReproFlake, without requiring new root-cause labels.

**Research question:** Does fine-tuning on prior flaky-test repairs improve the
validated repair rate compared with the same agent workflow using the base model?
Evaluation will use held-out projects, repeated tests under the conditions that
trigger flakiness, and regression checks.

## Current status

The full-corpus data acquisition and exploratory analysis are complete.
[dataset_acquisition_eda.ipynb](./dataset_acquisition_eda.ipynb) documents data
cleaning, repair patterns, historical execution outcomes, and project-wise splits.
[reproflake_data.py](./reproflake_data.py) supports acquisition, parsing, and
reference-pair preparation. Model training, agent implementation, and independent
repair evaluation are planned; the EDA does not establish repair performance.

## Run the data analysis

Use Python 3.11+ and Git:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m jupyter lab dataset_acquisition_eda.ipynb
```

Select the environment's kernel and run all cells from this directory. The
notebook contains concise methodology, figures, quality audits, and computed
answers to the four data-summary questions.

### Complete-corpus acquisition

- **All metadata:** 1,115 cases, 300 archives, and project/commit/issue references.
- **Every source archive:** approximately **45.1 GB** across the three official
  individually downloadable collections.
- **Complete execution results:** approximately **2.61 GB**.
- Allow approximately **47.7 GB plus working space**. A disk-space check runs
  before acquisition; all raw data remains compressed.
- Downloads use at most four workers and resume `.part` files with validated
  HTTP ranges. Published sizes and MD5 checksums are verified before promotion;
  SHA-256 fingerprints and timestamps are retained in the download manifest.
  Cached archives are reverified on subsequent runs.
- The default notebook analyzes the **full corpus**, without artifact sampling
  or a 512 MB transfer cap. Downloaded code/scripts are never executed.

The alternative approximately 44 GB `data.zip` is a nested bundle of the source
archives, not additional data. It is not downloaded a second time.

## Outputs

Complete raw archives, metadata, and inspected original/patch bytes are retained
in ignored `data/reproflake/`. Tables, figures, split assignments, provenance,
and a report-ready `data_summary.md` are saved in ignored `outputs/reproflake/`.
The executed notebook retains its results.

The expanded EDA includes:

- Row-level normalization examples and before/after cohort counts.
- Missingness by mechanism and discovery source.
- Repair size, edit scope, and context-size distributions.
- Paired original/fixed comparisons with case and observation weighting.
- Correlation sensitivity checks and project-wise split composition.

Explanations and interpretations are written in Markdown cells. The report text
is an authored summary of the analyzed snapshot, not Python-generated prose.
Rerunning refreshes tables and figures; review the written findings if the data
snapshot changes.

Reference source/patch JSONL files are separated by project partition under
`data/reproflake/reference_pairs/`. These are **reference/training-preparation
pairs, not a turnkey SFT dataset**: source context was selected from patch paths
and therefore reveals edit locations. Do not feed this oracle context to
evaluation agents. Actual training preparation still requires tokenizer-based
budgeting, license/sensitive-content checks, and appropriate prompt construction.

The blind holdout manifest omits solution patches, fixed revisions, issue/PR
links, and patch-derived metrics. Trusted benchmark preparation must mount only
original source and permitted execution evidence, never the full archive or
reference outputs. Derive evaluation context independently from the failing
test and original repository.

## Dataset and attribution

ReproFlake, *A Dataset of Reproducible Flaky-Test Failures*:

- [Paper](https://arxiv.org/abs/2605.21677)
- [Official website](https://sites.google.com/view/reproflakedataset)
- [Artifact documentation](https://anonymous.4open.science/r/ReproFlake-C9E6/README.md)
- [Complete dataset](https://doi.org/10.5281/zenodo.17508466): approximately **44 GB**.
- Individually accessible archives:
  [part 1](https://zenodo.org/records/18474558),
  [part 2](https://zenodo.org/records/18634501),
  [part 3](https://zenodo.org/records/18605131).
- [Execution results](https://zenodo.org/records/18451345): approximately **2.61 GB**.

Zenodo declares **CC BY 4.0**. Cite the paper and dataset, and review the separate
licenses of included project code before training or redistribution. Upstream
metadata is mutable; cached raw bytes, URLs, timestamps, and SHA-256 fingerprints
identify the analyzed snapshot. Zenodo file checksums are retained as publisher
provenance and verified for every downloaded archive. Parsed source/patch ZIP
members are additionally CRC-checked and locally hashed.

The full execution-results ZIP is downloaded and streamed with libarchive rather
than materializing its approximately 356 MB central directory as Python objects.
All result entries contribute to inventories; every `.log` payload is read for
line and explicit UTF-8 replacement counts. Repetition summaries support individual
test rows, embedded repeated invocations, and NonDex XML-report rows. Report-level
outcomes remain separate from exactly named test outcomes. Locale thousands
separators and condition typos are normalized; negative duration sentinels become
missing without losing valid outcomes. Malformed, unmatched, skipped/unknown,
and duplicate evidence is audited. Original/fixed/order/code-change conditions
remain separate. Historical publisher trials are not new independent validation
or evidence of this project's agent performance.

## Interpretation and leakage controls

- Test instances, source archives, projects, and unique repairs are different
  units. Do not turn multiple tests sharing one fix into independent examples.
- Approximately 70/15/15% of projects define training/validation/test partitions.
  Shared parsed patches crossing partitions are quarantined. Exact checks cover
  all parsed source archives; near-duplicate and fork checks remain necessary.
- Normalize documented spelling/whitespace and known duplicated-owner URL
  errors; preserve raw data, missing revisions, and structural missingness.
- Source and patch sizes inform context/output complexity. Patch-derived
  quantities, fixing revisions, and solution-bearing issues are not blind inputs.
- Removed assertion lines are review signals, not proof of weakening; skipped
  or deleted tests are conservatively quarantined from pair exports.
- Binary differences and generated build-output edits are retained in corpus
  statistics but excluded from ordinary text-repair exports. Large snapshot
  diffs are not silently truncated; inspect target quality and tokenizer budgets
  before converting the reference pairs into SFT examples.
- `git apply --check` is a static consistency check in a temporary directory:
  it does **not** establish compilation, behavioral correctness, or flakiness
  elimination. No downloaded tests, scripts, or containers are run.

## Validation

```bash
python -m unittest discover -s tests -v
python -m compileall -q reproflake_data.py tests
python -m black --check reproflake_data.py tests/test_reproflake_data.py dataset_acquisition_eda.ipynb
```

For headless execution with the existing dependencies:

```bash
python - <<'PY'
from pathlib import Path
import nbformat
from nbclient import NotebookClient

path = Path("dataset_acquisition_eda.ipynb")
notebook = nbformat.read(path, as_version=4)
NotebookClient(
    notebook, timeout=None, kernel_name="python3",
    resources={"metadata": {"path": str(Path.cwd())}},
).execute()
nbformat.write(notebook, path)
PY
```
