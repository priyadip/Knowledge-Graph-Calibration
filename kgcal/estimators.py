"""Intrinsic-dimension estimators of the width check (the width rule itself is model_ops.gride).

All estimators work on neighbour-distance ratios mu_i = r_{i,n2} / r_{i,n1} with n1 = k, n2 = 2k.

- code:       the estimator in kgcal/model_ops.py, d_k = 1 / mean(log mu), median over k.
- corrected:  d_k = (psi(2k) - psi(k)) / mean(log mu). Unbiased in expectation of log mu under a locally uniform
              Poisson model, since E[log mu] = (psi(n2) - psi(n1)) / d.
- gride:      maximum-likelihood GRIDE (Denti et al. 2022, Sci. Rep. 12:20005). Under the same model
              f(mu | d) = d (mu^d - 1)^(n2-n1-1) / (B(n2-n1, n1) mu^(d(n2-1)+1)), maximised over d.
- twonn:      the k = 1 case (Facco et al. 2017), d = 1 / mean(log(r2 / r1)).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import digamma

KS = np.unique(np.geomspace(1, 16, num=8).astype(int))      # identical to kgcal.model_ops.gride: [1 2 3 4 7 10 16]


def knn_dist(X: np.ndarray, kmax: int, device: str = "cuda") -> np.ndarray:
    """Sorted distances to the kmax nearest neighbours (self excluded), exact, float64 on GPU.

    Returns D in R^{P x kmax}; column j-1 is the distance to the j-th neighbour.
    """
    import torch                         # imported here so the dadapy venv (no torch) can use the estimators
    Xt = torch.as_tensor(X, dtype=torch.float64, device=device)
    sq = (Xt * Xt).sum(1)
    out = []
    for s in range(0, Xt.shape[0], 1024):
        blk = Xt[s:s + 1024]
        d2 = sq[s:s + 1024, None] + sq[None, :] - 2 * blk @ Xt.T      # squared distances, (b, P)
        d2.clamp_(min=0)
        d2[torch.arange(blk.shape[0]), torch.arange(s, s + blk.shape[0])] = float("inf")   # drop self
        out.append(torch.topk(d2, kmax, dim=1, largest=False).values.sqrt().cpu())
    return torch.cat(out).numpy()


def ratios(D: np.ndarray, k: int) -> np.ndarray:
    r1, r2 = D[:, k - 1], D[:, 2 * k - 1]
    v = r1 > 0
    mu = r2[v] / r1[v]
    return mu[mu > 1]


def d_code(mu: np.ndarray) -> float:
    return 1.0 / np.mean(np.log(mu)) if len(mu) else np.nan


def d_corrected(mu: np.ndarray, k: int) -> float:
    return (digamma(2 * k) - digamma(k)) / np.mean(np.log(mu)) if len(mu) else np.nan


def d_gride(mu: np.ndarray, k: int) -> float:
    """MLE of d for ratios of neighbour orders n1 = k, n2 = 2k."""
    if not len(mu):
        return np.nan
    n1, n2 = k, 2 * k
    lm = np.log(mu)
    S = lm.sum()
    n = len(mu)

    def nll(logd):
        d = np.exp(logd)
        # log(mu^d - 1) = d log mu + log(1 - mu^-d), stable for large d log mu
        t = (n2 - n1 - 1) * np.sum(d * lm + np.log1p(-np.exp(-d * lm))) if n2 - n1 - 1 else 0.0
        return -(n * np.log(d) + t - (d * (n2 - 1) + 1) * S)

    res = minimize_scalar(nll, bounds=(np.log(0.05), np.log(5000.0)), method="bounded", options={"xatol": 1e-9})
    return float(np.exp(res.x))


def all_estimates(D: np.ndarray, ks=KS) -> dict:
    """Per-scale estimates and the median over scales, for each estimator."""
    per = {"code": [], "corrected": [], "gride": []}
    for k in ks:
        mu = ratios(D, int(k))
        per["code"].append(d_code(mu))
        per["corrected"].append(d_corrected(mu, int(k)))
        per["gride"].append(d_gride(mu, int(k)))
    out = {"ks": [int(k) for k in ks], "per_k": per}
    for name, v in per.items():
        v = np.asarray(v, float)
        v = v[np.isfinite(v) & (v > 0)]
        out[name] = float(np.median(v)) if v.size else float("nan")
    out["twonn"] = per["code"][0]
    return out
