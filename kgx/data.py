"""Data for the further families: kgcal.data unchanged, with the paired calibration texts.

True texts (KG-D, KG-QA, KG-O2, KG-Ac, WIKI) come from data/calib and the variants from data/calib_paired: line i of
KG-D~ and of KG-S is the variant of line i of KG-D, and line i of KG-QA~ is the variant of line i of KG-QA. The PopQA
split is data/popqa, so the facts, entities and in-context examples are those of Qwen2.5 and OLMo-2.
"""
from __future__ import annotations

import random

import torch

from kgcal import data as KD
from kgcal.data import ans, c4_lines, cloze, declarative, facts_by_split, load_popqa, make_split  # noqa: F401

from kgcal import paths as P

RESULTS = P.FAMILIES


def calib_path(name: str):
    """Calibration text `name`: every family run uses the paired variants."""
    return P.calib_file(name, paired=True)


# the true text each paired variant was written from
PARTNER = {"KG-D~": "KG-D", "KG-S": "KG-D", "KG-QA~": "KG-QA"}


def pack(tok, lines: list[str], n_seq: int, seq_len: int = 512, bos: int | None = None) -> torch.Tensor:
    """kgcal.data.pack; with a BOS family every sequence is BOS followed by seq_len - 1 text tokens."""
    if bos is None:
        return KD.pack(tok, lines, n_seq, seq_len)
    body = seq_len - 1
    ids: list[int] = []
    for t in lines:
        ids.extend(tok(t + "\n", add_special_tokens=False)["input_ids"])
        if len(ids) >= n_seq * body:
            break
    if len(ids) < n_seq * body:
        raise ValueError(f"not enough text: {len(ids)} tokens for {n_seq} x {body}")
    x = torch.tensor(ids[: n_seq * body]).view(n_seq, body)
    return torch.cat([torch.full((n_seq, 1), bos, dtype=x.dtype), x], 1)


def paired_shuffle(srcs: dict[str, list[str]], cseed: int) -> None:
    """Calibration-sampling seed: shuffle every source in place, keeping each variant paired with its true text.

    The permutations are drawn exactly as kgcal.run_grid draws them (one random.Random(cseed), sources in sorted
    order, one shuffle each), then every variant takes the permutation its true partner received. This is the paired
    resampling of scripts/experiments/paired_variants.py, for any grid.
    """
    rng = random.Random(cseed)
    perms = {}
    for k in sorted(srcs):
        p = list(range(len(srcs[k])))
        rng.shuffle(p)
        perms[k] = p
    for k, partner in PARTNER.items():
        if k in srcs:
            if partner not in perms:                   # partner not in this grid: draw its permutation the same way
                raise ValueError(f"{k} is resampled only together with {partner}")
            assert len(srcs[k]) == len(srcs[partner]), (k, partner)
            perms[k] = perms[partner]
    for k in srcs:
        srcs[k] = [srcs[k][i] for i in perms[k]]
