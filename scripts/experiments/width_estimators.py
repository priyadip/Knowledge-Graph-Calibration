"""Width check on Qwen2.5 and OLMo-2: intrinsic dimension of real FFN activation clouds, all estimators, and
sensitivity to sample size.

Usage:  python scripts/experiments/width_estimators.py qwen15b
1. Rebuild the exact calibration half-set C4a and activation clouds used by run_grid (seed 0) and check that the code
   estimator reproduces results/grid/<tag>_gride.json.
2. On the same 2,200-unit subsample: code, corrected, GRIDE-MLE and TWO-NN estimates per layer.
3. Sample size: clouds with 8,192 tokens; prefixes of 512 ... 8,192 tokens x 1,100 / 2,200 / 4,400 / all units;
   plus three unit-subsample seeds at the paper setting (2,048 tokens, 2,200 units).
Writes results/width_check/<tag>_dims.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from kgcal import paths as P            # noqa: E402
from kgcal import data as D            # noqa: E402
from kgcal import model_ops as MO      # noqa: E402
from kgcal.estimators import KS, knn_dist, all_estimates   # noqa: E402

MODELS = {"qwen15b": "Qwen/Qwen2.5-1.5B", "olmo1b": "allenai/OLMo-2-0425-1B",
          "qwen7b": "Qwen/Qwen2.5-7B", "olmo7b": "allenai/OLMo-2-1124-7B"}
tag = sys.argv[1]
t0 = time.time()
log = lambda *a: print(time.strftime("%H:%M:%S"), f"[{tag}]", *a, flush=True)

model, tok = MO.load(MODELS[tag])
c4 = D.c4_lines(3000)
C4a = D.pack(tok, c4[:2500], 128)[:64]                  # exactly run_grid's half["C4a"] for cseed 0, half 64
N = model.config.intermediate_size

# ---- 1. reproduction of the cached dimensions
clouds = MO.collect_clouds(model, C4a)                  # n_rows 2048, seed 0, as run_grid
dims_repro = MO.gride_dims(clouds)                      # sklearn path, exactly the main code
cached = json.loads((P.GRID / f"{tag}_gride.json").read_text())["dims"]
gap = float(np.max(np.abs(np.array(dims_repro) - np.array(cached))))
log(f"reproduction: max |d_repro - d_cached| = {gap:.3g}")

# ---- 2. all estimators on the identical subsample (same rng as gride_dims)
rng = np.random.default_rng(0)
layers = []
for l, c in enumerate(clouds):
    X = c.T.numpy().astype(np.float64)
    X = X[rng.choice(X.shape[0], min(2200, X.shape[0]), replace=False)]
    Dm = knn_dist(X, 2 * int(KS[-1]))
    est = all_estimates(Dm)
    est["code_sklearn"] = dims_repro[l]
    layers.append(est)
log("estimators done:", [(round(e["code"], 1), round(e["gride"], 1)) for e in layers[:4]], "...")

# unit-subsample seeds at the paper setting
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
del clouds

# ---- 3. sample size
big = MO.collect_clouds(model, C4a, n_rows=8192, seed=1)
sweep = []
for l, c in enumerate(big):
    for n_tok in (512, 1024, 2048, 4096, 8192):
        Xfull = c[:n_tok].T.numpy().astype(np.float64)          # units x tokens
        for n_pts in (1100, 2200, 4400, N):
            r = np.random.default_rng(0)
            X = Xfull[r.choice(N, min(n_pts, N), replace=False)]
            e = all_estimates(knn_dist(X, 32))
            sweep.append({"layer": l, "n_tok": n_tok, "n_pts": int(min(n_pts, N)),
                          **{k: e[k] for k in ("code", "corrected", "gride", "twonn")}})
    log(f"sweep layer {l} done ({time.time() - t0:.0f}s)")

out = {"tag": tag, "model": MODELS[tag], "N": N, "L": len(layers), "repro_gap": gap, "cached": cached,
       "layers": layers, "subsample_seeds": seeds, "sweep": sweep}
(P.WIDTHS / f"{tag}_dims.json").write_text(json.dumps(out))
log(f"all done in {time.time() - t0:.0f}s")
