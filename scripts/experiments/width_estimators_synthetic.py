"""Width check: the estimators on point clouds of known intrinsic dimension.

Three families, each with P = 2,200 points (the paper's subsample size), embedded in R^2048 by a random orthonormal map:
  cube:   uniform in [0,1]^d
  gauss:  standard normal in R^d
  sphere: uniform on the unit sphere S^d (intrinsic dimension d) in R^{d+1}
d in {2, 4, 8, 16, 32, 64}, 3 seeds.
python scripts/experiments/width_estimators_synthetic.py  ->  results/width_check/e1_synthetic.json (CPU only)
"""
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from kgcal import paths as P            # noqa: E402
from kgcal.estimators import KS, knn_dist, all_estimates   # noqa: E402
P, AMB = 2200, 2048
rows = []
for fam in ("cube", "gauss", "sphere"):
    for d in (2, 4, 8, 16, 32, 64):
        for seed in range(3):
            rng = np.random.default_rng(seed)
            if fam == "cube":
                Z = rng.random((P, d))
            elif fam == "gauss":
                Z = rng.standard_normal((P, d))
            else:
                Z = rng.standard_normal((P, d + 1))
                Z /= np.linalg.norm(Z, axis=1, keepdims=True)
            Q, _ = np.linalg.qr(rng.standard_normal((AMB, Z.shape[1])))      # orthonormal columns
            X = Z @ Q.T
            e = all_estimates(knn_dist(X, 2 * int(KS[-1])))
            rows.append({"family": fam, "d": d, "seed": seed, **{k: e[k] for k in ("code", "corrected", "gride", "twonn")},
                         "per_k": e["per_k"]})
            print(fam, d, seed, {k: round(e[k], 2) for k in ("code", "corrected", "gride", "twonn")}, flush=True)
(P.WIDTHS / "e1_synthetic.json").write_text(json.dumps({"ks": [int(k) for k in KS], "rows": rows}))
