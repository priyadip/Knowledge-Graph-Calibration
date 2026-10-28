"""Selection rules, consumers (none, bias-only, ridge), and the per-layer terms of Lemma 1."""
from __future__ import annotations

import torch


def select(rule: str, W: torch.Tensor, M: int, M2: torch.Tensor | None = None, dense_layer: dict | None = None,
           seed: int = 0) -> torch.Tensor:
    """Indices of the M kept units (sorted). W is down_proj (d x N)."""
    N = W.shape[1]
    if rule == "actnorm":                       # Wanda-style: ||W[:, j]|| * RMS_Q(h_j)
        score = W.float().norm(dim=0) * torch.sqrt(torch.diag(M2)[:N].float().clamp(min=0))
    elif rule == "magnitude":                   # calibration-free: product of the unit's three weight norms
        score = (W.float().norm(dim=0) * dense_layer["gate_proj"].float().norm(dim=1)
                 * dense_layer["up_proj"].float().norm(dim=1))
    elif rule == "random":
        score = torch.rand(N, generator=torch.Generator().manual_seed(seed))
    else:
        raise ValueError(rule)
    return torch.argsort(score, descending=True)[:M].sort().values


def zidx(S: torch.Tensor, N: int) -> torch.Tensor:
    return torch.cat([S, torch.tensor([N])])


def consumer_none(W: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    A = torch.zeros(len(S) + 1, W.shape[0], dtype=torch.float64)
    A[:-1] = W[:, S].T.double()
    return A


def bias_only(W: torch.Tensor, S: torch.Tensor, M2: torch.Tensor) -> torch.Tensor:
    """FLAP-style compensation: add W_{S^c} E_Q[h_{S^c}] as a bias; kept weights unchanged."""
    N = W.shape[1]
    keep = torch.zeros(N, dtype=torch.bool)
    keep[S] = True
    mu = M2[:N, N].double()                    # E_Q[h] sits in the last column of the moments of [h, 1]
    return (W[:, ~keep].double() @ mu[~keep])


def ridge(W: torch.Tensor, S: torch.Tensor, M2: torch.Tensor, lam_rel: float = 1e-4,
          device: str = "cuda") -> torch.Tensor:
    """argmin_A E_Q ||W h - A^T z||^2 + lam ||A_{1:M}||^2 (intercept not shrunk), solved in float64."""
    N = W.shape[1]
    zi = zidx(S, N).to(device)
    M2d = M2.to(device)
    Szz = M2d[zi][:, zi].double()
    Szh = M2d[zi][:, :N].double()
    m = len(S)
    if lam_rel > 0:
        Szz[range(m), range(m)] += lam_rel * torch.diag(Szz)[:m].mean()
    else:                                      # numerical floor only, keeps the normal equations to ~1e-10
        Szz[range(m + 1), range(m + 1)] += 1e-10 * torch.diag(Szz).mean()
    rhs = Szh @ W.to(device).double().T
    return torch.linalg.solve(Szz, rhs).cpu()


def error(M2: torch.Tensor, W: torch.Tensor, S: torch.Tensor, A: torch.Tensor, device: str = "cuda") -> float:
    """E_P ||W h - A^T z||^2 from the moments of [h, 1] under P."""
    N = W.shape[1]
    zi = zidx(S, N).to(device)
    M2d = M2.to(device).double()
    Wd = W.to(device).double()
    Ad = A.to(device).double()
    Shh, Szh, Szz = M2d[:N, :N], M2d[zi][:, :N], M2d[zi][:, zi]
    return float(torch.trace(Wd @ Shh @ Wd.T) - 2 * torch.trace(Ad.T @ Szh @ Wd.T) + torch.trace(Ad.T @ Szz @ Ad))


def lemma_terms(M2_fit: torch.Tensor, M2_eval: torch.Tensor, W: torch.Tensor, S: torch.Tensor,
                consumers: dict[str, torch.Tensor], device: str = "cuda") -> dict:
    """Normalized Lemma 1 quantities on a target P.

    The P-optimal consumer is fitted on M2_fit and evaluated on M2_eval (two halves of P: cross-fitting), which
    gives an honest estimate of the unavoidable fraction. For each consumer: residual ratio r = err / discarded,
    mismatch fraction = r - unavoidable fraction. Also the exact in-sample identity gap on M2_eval.
    """
    if len(S) == W.shape[1]:                   # layer kept whole by the width rule: nothing is discarded
        return None
    A_P = ridge(W, S, M2_fit, lam_rel=1e-6, device=device)
    none = consumer_none(W, S)
    base = error(M2_eval, W, S, none, device)
    u = error(M2_eval, W, S, A_P, device) / base
    out = {"unavoidable": u}
    A_P_in = ridge(W, S, M2_eval, lam_rel=0.0, device=device)
    eps_in = error(M2_eval, W, S, A_P_in, device)
    N = W.shape[1]
    zi = zidx(S, N)
    Szz = M2_eval[zi][:, zi].double()
    for name, A in consumers.items():
        e = error(M2_eval, W, S, A, device)
        dA = (A_P_in - A).double()
        mism_in = float(torch.trace(dA.T @ Szz @ dA))
        out[name] = {"ratio": e / base, "mismatch": e / base - u, "identity_gap": abs(e - eps_in - mism_in) / e}
    return out
