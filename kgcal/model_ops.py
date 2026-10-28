"""Model access for SwiGLU decoders (Qwen2, OLMo2): FFN statistics by layer groups, GRIDE widths, apply/restore."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors


class StopForward(Exception):
    pass


# Hugging Face revisions of the four checkpoints (the snapshots every run of the paper loaded)
REVISIONS = {"Qwen/Qwen2.5-1.5B": "8faed761d45a263340a0528343f099c05c9a4323",
             "allenai/OLMo-2-0425-1B": "a1847dff35000b4271fa70afc5db10fd29fedbdf",
             "Qwen/Qwen2.5-7B": "d149729398750b98c0af14eb82c78cfe92750796",
             "allenai/OLMo-2-1124-7B": "7df9a82518afdecae4e8c026b27adccc8c1f0032"}


def load(name: str, device: str = "cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    rev = REVISIONS.get(name)
    tok = AutoTokenizer.from_pretrained(name, revision=rev)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, revision=rev, dtype=torch.bfloat16).to(device).eval()
    return model, tok


def mlps(model):
    return [layer.mlp for layer in model.model.layers]


def dense_copy(model) -> list[dict]:
    """CPU copy of every FFN weight, used to rebuild any compressed configuration."""
    return [{k: getattr(m, k).weight.detach().cpu().clone() for k in ("gate_proj", "up_proj", "down_proj")}
            for m in mlps(model)]


@torch.no_grad()
def collect_moments(model, ids: torch.Tensor, layers: list[int], batch: int = 8) -> dict[int, torch.Tensor]:
    """Uncentered second moments E[z z^T] of z = [h, 1] at the down_proj input of each layer in `layers`.

    Forward passes stop after the deepest requested layer. Returns fp32 CPU tensors of shape (N+1, N+1).
    """
    dev = next(model.parameters()).device
    N = model.config.intermediate_size
    acc = {i: torch.zeros(N + 1, N + 1, device=dev) for i in layers}
    last = max(layers)
    hooks = []
    for i in layers:
        def hk(mod, inp, out, i=i):
            h = inp[0].reshape(-1, N).float()
            z = torch.cat([h, torch.ones(h.shape[0], 1, device=dev)], 1)
            acc[i].addmm_(z.T, z)
            if i == last:
                raise StopForward
        hooks.append(mlps(model)[i].down_proj.register_forward_hook(hk))
    try:
        for b in range(0, ids.shape[0], batch):
            try:
                model(input_ids=ids[b:b + batch].to(dev))
            except StopForward:
                pass
    finally:
        for h in hooks:
            h.remove()
    n = float(ids.numel())
    return {i: (acc[i] / n).cpu() for i in layers}


@torch.no_grad()
def collect_clouds(model, ids: torch.Tensor, n_rows: int = 2048, batch: int = 8, seed: int = 0) -> list[torch.Tensor]:
    """For GRIDE: n_rows random token activations per layer (rows) over all FFN units (columns)."""
    dev = next(model.parameters()).device
    N = model.config.intermediate_size
    L = model.config.num_hidden_layers
    g = torch.Generator(device="cpu").manual_seed(seed)
    per_batch = max(1, n_rows // max(1, ids.shape[0] // batch))
    store = [[] for _ in range(L)]
    hooks = []
    for i in range(L):
        def hk(mod, inp, out, i=i):
            h = inp[0].reshape(-1, N)
            idx = torch.randperm(h.shape[0], generator=g)[:per_batch].to(h.device)
            store[i].append(h[idx].float().cpu())
        hooks.append(mlps(model)[i].down_proj.register_forward_hook(hk))
    try:
        for b in range(0, ids.shape[0], batch):
            model(input_ids=ids[b:b + batch].to(dev))
    finally:
        for h in hooks:
            h.remove()
    return [torch.cat(s)[:n_rows] for s in store]


def gride(X: np.ndarray, n_neighbors: int = 16, n_iters: int = 8) -> float:
    """GRIDE (Denti et al. 2022): median over neighbour scales of 1 / mean(log(r_2k / r_k))."""
    ks = np.unique(np.geomspace(1, n_neighbors, num=n_iters).astype(int))
    cap = int(2 * ks[-1])
    dist, _ = NearestNeighbors(n_neighbors=cap + 1).fit(X).kneighbors(X)
    est = []
    for k in ks:
        r1, r2 = dist[:, k], dist[:, 2 * k]
        v = r1 > 0
        mu = r2[v] / r1[v]
        mu = mu[mu > 1]
        est.append(1.0 / np.mean(np.log(mu)) if len(mu) else np.nan)
    est = np.asarray(est, dtype=float)
    est = est[np.isfinite(est) & (est > 0)]
    if est.size == 0:
        raise RuntimeError("GRIDE returned no finite estimate")
    return float(np.median(est))


def gride_dims(clouds: list[torch.Tensor], max_points: int = 2200, seed: int = 0) -> list[float]:
    """Neuron cloud: each unit is a point, its activations over calibration tokens are its coordinates."""
    rng = np.random.default_rng(seed)
    dims = []
    for c in clouds:
        X = c.T.numpy().astype(np.float64)
        X = X[rng.choice(X.shape[0], min(max_points, X.shape[0]), replace=False)]
        dims.append(gride(X))
    return dims


def widths(dims: list[float], alpha: float, N: int, m_min: int = 4) -> list[int]:
    """The width rule: M_l = round(alpha * d_l), clipped to [m_min, N]."""
    return [int(np.clip(round(alpha * d), m_min, N)) for d in dims]


def alpha_for_fraction(dims: list[float], N: int, target: float) -> float:
    """One alpha per network such that the kept FFN fraction is closest to `target` (alpha rounded to 0.1)."""
    lo, hi = 0.1, N / min(dims) + 1
    for _ in range(60):
        mid = (lo + hi) / 2
        frac = sum(widths(dims, mid, N)) / (len(dims) * N)
        lo, hi = (mid, hi) if frac < target else (lo, mid)
    return round((lo + hi) / 2, 1)


def apply(model, dense: list[dict], i: int, S: torch.Tensor, A: torch.Tensor | None = None,
          bias: torch.Tensor | None = None):
    """Replace FFN i by its compressed version: kept units S, consumer A ((M+1) x d) or original weights.

    A is None: keep W_S (no repair), optionally with a bias vector (bias-only repair).
    """
    dev = next(model.parameters()).device
    m = mlps(model)[i]
    d, M = dense[i]["down_proj"].shape[0], len(S)
    Sc = S.cpu()
    g = nn.Linear(d, M, bias=False, device=dev, dtype=torch.bfloat16)
    u = nn.Linear(d, M, bias=False, device=dev, dtype=torch.bfloat16)
    g.weight.data = dense[i]["gate_proj"][Sc].to(dev)
    u.weight.data = dense[i]["up_proj"][Sc].to(dev)
    has_bias = A is not None or bias is not None
    dn = nn.Linear(M, d, bias=has_bias, device=dev, dtype=torch.bfloat16)
    if A is None:
        dn.weight.data = dense[i]["down_proj"][:, Sc].to(dev)
        if bias is not None:
            dn.bias.data = bias.to(dev, torch.bfloat16)
    else:
        dn.weight.data = A[:M].T.contiguous().to(dev, torch.bfloat16)
        dn.bias.data = A[M].to(dev, torch.bfloat16)
    m.gate_proj, m.up_proj, m.down_proj = g, u, dn


def restore(model, dense: list[dict]):
    dev = next(model.parameters()).device
    for i, m in enumerate(mlps(model)):
        for k in ("gate_proj", "up_proj", "down_proj"):
            w = dense[i][k]
            lin = nn.Linear(w.shape[1], w.shape[0], bias=False, device=dev, dtype=torch.bfloat16)
            lin.weight.data = w.to(dev)
            setattr(m, k, lin)
