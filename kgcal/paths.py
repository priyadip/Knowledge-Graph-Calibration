"""Locations of every input and output, relative to the repository root.

Set KGCAL_ROOT to read data/ and results/ from another copy of the repository. Models and datasets are loaded through
the Hugging Face cache (HF_HOME), as usual.
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get("KGCAL_ROOT", Path(__file__).resolve().parents[1]))
DATA, RESULTS, OUTPUTS = ROOT / "data", ROOT / "results", ROOT / "outputs"

# inputs (data/README.md)
SPLITS = DATA / "popqa"                  # subject-disjoint PopQA split and leakage checks
KG = DATA / "wikidata"                   # Wikidata statements, property labels and types; Wikipedia lead paragraphs
CALIB = DATA / "calib"                   # calibration texts built by kgcal.data
CALIB_PAIRED = DATA / "calib_paired"     # KG-D~, KG-QA~ and KG-S: line i is the variant of line i of KG-D or KG-QA
CALIB_EQ = DATA / "calib_eq"             # calibration texts written from the EntityQuestions graph
EQ = DATA / "entityquestions"            # EntityQuestions test sample; raw/ holds the downloaded dataset
CONTROLS = DATA / "controls"             # matched-format facts, truth variants, multiple-choice candidates

# per-fact results of the runs (results/README.md)
GRID = RESULTS / "grid"                  # kgcal.run_grid on Qwen2.5 and OLMo-2
PAIRED = RESULTS / "paired"              # paired-variant reruns of the counterfactual and synthetic-subject conditions
MERGED = RESULTS / "merged"              # Qwen2.5 and OLMo-2 results the paper reports (merge_results.py)
FAMILIES = RESULTS / "families"          # kgx.run_grid on Llama-3.1, Mistral and Phi-4
WIDTHS = RESULTS / "width_check"         # width estimators and runs at alternative widths
EQ_RESULTS = RESULTS / "entityquestions"
CONTROL_RESULTS = RESULTS / "controls"
MIXTURE = RESULTS / "mixture"
SEQUENTIAL = RESULTS / "sequential"
OBS = RESULTS / "obs"
SPARSEGPT = RESULTS / "sparsegpt"
PAIRED_CLOZE = RESULTS / "paired_cloze"
ANSWERS = RESULTS / "answers"
MECHANISM = RESULTS / "mechanism"
COST = RESULTS / "cost"

# written by the analysis
TABLES, FIGURES, REPORTS = OUTPUTS / "tables", OUTPUTS / "figures", OUTPUTS / "reports"
for _d in (TABLES, FIGURES, REPORTS):
    _d.mkdir(parents=True, exist_ok=True)


def calib_file(name: str, paired: bool = True) -> Path:
    """Calibration text `name` (KG-D, KG-QA~, ...): the paired variant when one exists and paired is True.

    The first grid of Qwen2.5 and OLMo-2 (results/grid) used independently shuffled variants (paired=False); every
    result the paper reports for KG-D~, KG-QA~ and KG-S comes from the paired files."""
    p = CALIB_PAIRED / f"{name}.jsonl"
    return p if paired and p.exists() else CALIB / f"{name}.jsonl"
