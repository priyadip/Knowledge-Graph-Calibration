"""MIX(KG-S)/MIX(KG-S) with KG-S paired line by line with KG-D (same relations and objects, made-up subjects).

Usage:  python scripts/experiments/paired_subjects.py <tag> [run_grid options, e.g. --batch 64 --group 2]
Only MIX:KG-S|MIX:KG-S is run (and the dense model, which run_grid always scores), at the main widths, calibration
sample 0 and the generation batch of the main run (96 for the 1B models, 64 for the 7B models); MIX:KG-D|MIX:KG-D is
taken from the main run. Writes results/paired/a4_entities/<tag>_ent.jsonl and <tag>_ent_bits.json.
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
new_tag = f"{tag}_ent"
out = P.PAIRED / "a4_entities"
out.mkdir(parents=True, exist_ok=True)
shutil.copy(P.GRID / f"{tag}_gride.json", out / f"{new_tag}_gride.json")
orig_grid = R.grid
R.grid = lambda name: [c for c in orig_grid("core") if c[0] == "MIX:KG-S|MIX:KG-S"]
D.PAIRED = True
D.RESULTS = out
sys.argv = ["run_grid", "--model", MODELS[tag], "--tag", new_tag, "--grid", "core", "--no_cloze"] + extra
R.main()
