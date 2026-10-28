"""Width check for one model (scripts/experiments/width_estimators.py for Qwen2.5 and OLMo-2, without the
sample-size sweep).

Usage (from the repository root):  python -m kgx.widths llama8b
1. Rebuild the calibration half-set C4a and the activation clouds of run_grid (seed 0) and check that the width
   estimator reproduces results/families/<tag>_gride.json.
2. On the same 2,200-unit subsample of every layer: the code estimator, its corrected form, GRIDE-MLE and TWO-NN
   (kgcal.estimators), and three further unit-subsample seeds.
Writes results/families/<tag>_dims.json and <tag>_gmle_dims.json (GRIDE-MLE dimensions, for the run with those widths).
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

from kgcal.estimators import KS, all_estimates, knn_dist

from . import data as D
from . import model_ops as MO
from .models import load, text_config


tag = sys.argv[1]
t0 = time.time()
log = lambda *a: print(time.strftime("%H:%M:%S"), f"[{tag}]", *a, flush=True)

model, tok, meta = load(tag)
c4 = D.c4_lines(3000)
C4a = D.pack(tok, c4[:2500], 128, bos=meta["bos"])[:64]      # exactly run_grid's half["C4a"] for cseed 0, half 64
N = text_config(model).intermediate_size

clouds = MO.collect_clouds(model, C4a)                       # n_rows 2048, seed 0, as run_grid
dims_repro = MO.gride_dims(clouds)
cached = json.loads((D.RESULTS / f"{tag}_gride.json").read_text())["dims"]
gap = float(np.max(np.abs(np.array(dims_repro) - np.array(cached))))
log(f"reproduction: max |d_repro - d_cached| = {gap:.3g}")

rng = np.random.default_rng(0)                              # the subsample of model_ops.gride_dims
layers = []
for c in clouds:
    X = c.T.numpy().astype(np.float64)
    X = X[rng.choice(X.shape[0], min(2200, X.shape[0]), replace=False)]
    est = all_estimates(knn_dist(X, 2 * int(KS[-1])))
    layers.append(est)
for est, d in zip(layers, dims_repro):
    est["code_sklearn"] = d
log("estimators done:", [(round(e["code"], 1), round(e["gride"], 1)) for e in layers[:4]], "...")

seeds = {}
for s in (1, 2, 3):
    r = np.random.default_rng(s)
    seeds[s] = []
    for c in clouds:
        X = c.T.numpy().astype(np.float64)
        X = X[r.choice(X.shape[0], 2200, replace=False)]
        e = all_estimates(knn_dist(X, 32))
        seeds[s].append({k: e[k] for k in ("code", "corrected", "gride", "twonn")})
log("subsample seeds done")

out = {"tag": tag, "model": meta["repo"], "N": N, "L": len(layers), "repro_gap": gap, "cached": cached,
       "layers": layers, "subsample_seeds": seeds}
(D.RESULTS / f"{tag}_dims.json").write_text(json.dumps(out))
(D.RESULTS / f"{tag}_gmle_dims.json").write_text(json.dumps({"dims": [e["gride"] for e in layers], "variant": "gmle"}))
log(f"all done in {time.time() - t0:.0f}s")
