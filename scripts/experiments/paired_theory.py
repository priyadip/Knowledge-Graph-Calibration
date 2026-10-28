"""Lemma 1 terms of the paired counterfactual selections (the consumers of the theory grid, measured on the dense model).

Usage:  python scripts/experiments/paired_theory.py <tag> [run_grid options, e.g. --group 2]
The theory runs of results/grid (--grid theory --theory_only, the main widths), with the paired files of
data/calib_paired, so MIX(KG-D~) and MIX(KG-QA~) hold the same facts as MIX(KG-D) and MIX(KG-QA).
Writes results/paired/a3_theory/<tag>_aligned_theory.jsonl; merge_results.py takes the rows of the counterfactual
selections from it.
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from kgcal import data as D            # noqa: E402
from kgcal import paths as P           # noqa: E402
from kgcal import run_grid as R        # noqa: E402

MODELS = {"qwen15b": "Qwen/Qwen2.5-1.5B", "olmo1b": "allenai/OLMo-2-0425-1B",
          "qwen7b": "Qwen/Qwen2.5-7B", "olmo7b": "allenai/OLMo-2-1124-7B"}
tag, extra = sys.argv[1], sys.argv[2:]
new_tag = f"{tag}_aligned"
out = P.PAIRED / "a3_theory"
out.mkdir(parents=True, exist_ok=True)
shutil.copy(P.GRID / f"{tag}_gride.json", out / f"{new_tag}_gride.json")
D.PAIRED = True
D.RESULTS = out
sys.argv = ["run_grid", "--model", MODELS[tag], "--tag", new_tag, "--grid", "theory", "--theory_only"] + extra
R.main()
