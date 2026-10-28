"""kgcal.model_ops for any family in models.REGISTRY.

The functions below are kgcal.model_ops with two changes only: the decoder layers are found with
models.decoder_layers (a multimodal checkpoint nests them one level deeper), and the sizes come from the text config.
The width rule, the GRIDE estimator and the cloud subsampling are imported unchanged.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from kgcal.model_ops import StopForward, alpha_for_fraction, gride, gride_dims, widths  # noqa: F401  (unchanged)

from .models import decoder_layers, text_config


def mlps(model):
    return [layer.mlp for layer in decoder_layers(model)]


def dense_copy(model) -> list[dict]:
    """CPU copy of every FFN weight, used to rebuild any compressed configuration."""
    return [{k: getattr(m, k).weight.detach().cpu().clone() for k in ("gate_proj", "up_proj", "down_proj")}
            for m in mlps(model)]


@torch.no_grad()
def collect_moments(model, ids: torch.Tensor, layers: list[int], batch: int = 8) -> dict[int, torch.Tensor]:
    """Uncentered second moments E[z z^T] of z = [h, 1] at the down_proj input of each layer in `layers`.

    Forward passes stop after the deepest requested layer. Returns fp32 CPU tensors of shape (N+1, N+1).
    """
    dev = model.get_input_embeddings().weight.device          # where the inputs go
    N = text_config(model).intermediate_size
    acc = {i: torch.zeros(N + 1, N + 1, device=mlps(model)[i].down_proj.weight.device) for i in layers}
    last = max(layers)
    hooks = []
    for i in layers:
        def hk(mod, inp, out, i=i):
            h = inp[0].reshape(-1, N).float()
            z = torch.cat([h, torch.ones(h.shape[0], 1, device=h.device)], 1)
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
    dev = model.get_input_embeddings().weight.device
    cfg = text_config(model)
    N, L = cfg.intermediate_size, cfg.num_hidden_layers
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


def apply(model, dense: list[dict], i: int, S: torch.Tensor, A: torch.Tensor | None = None,
          bias: torch.Tensor | None = None):
    """Replace FFN i by its compressed version: kept units S, consumer A ((M+1) x d) or original weights.

    A is None: keep W_S (no repair), optionally with a bias vector (bias-only repair). The activation is untouched.
    """
    m = mlps(model)[i]
    dev = m.down_proj.weight.device                           # the block's own GPU
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
    for i, m in enumerate(mlps(model)):
        dev = m.down_proj.weight.device
        for k in ("gate_proj", "up_proj", "down_proj"):
            w = dense[i][k]
            lin = nn.Linear(w.shape[1], w.shape[0], bias=False, device=dev, dtype=torch.bfloat16)
            lin.weight.data = w.to(dev)
            setattr(m, k, lin)
