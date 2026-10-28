"""Run kgcal.run_grid with alternative widths, writing every output under results/width_check.

Usage:
  python scripts/experiments/grid_alt_widths.py <tag> <variant> [run_grid options...]
variant: gmle   = widths from GRIDE-MLE d_l (paper setting cloud)
         seed1  = widths from the code estimator on another 2,200-unit subsample (noise baseline)
The width file results/width_check/<tag>_<variant>_gride.json is written first; run_grid then finds it under its tag,
so it does not recompute widths. run_grid's output folder D.RESULTS is pointed at results/width_check for this
process only.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from kgcal import paths as P            # noqa: E402
from kgcal import data as D            # noqa: E402
from kgcal import run_grid as R        # noqa: E402

MODELS = {"qwen15b": "Qwen/Qwen2.5-1.5B", "olmo1b": "allenai/OLMo-2-0425-1B",
          "qwen7b": "Qwen/Qwen2.5-7B", "olmo7b": "allenai/OLMo-2-1124-7B"}
tag, variant, extra = sys.argv[1], sys.argv[2], sys.argv[3:]
dims_file = json.loads((P.WIDTHS / f"{tag}_dims.json").read_text())
if variant == "gmle":
    dims = [e["gride"] for e in dims_file["layers"]]
elif variant == "seed1":
    dims = [x["code"] for x in dims_file["subsample_seeds"]["1"]]
else:
    raise ValueError(variant)
new_tag = f"{tag}_{variant}"
(P.WIDTHS / f"{new_tag}_gride.json").write_text(json.dumps({"model": MODELS[tag], "dims": dims, "variant": variant}))

D.RESULTS = P.WIDTHS
sys.argv = ["run_grid", "--model", MODELS[tag], "--tag", new_tag] + extra
R.main()
