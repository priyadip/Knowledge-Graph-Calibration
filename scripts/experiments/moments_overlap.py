"""Activation moments, kept units and repairs of paired calibration texts (the mechanism behind the truth result).

Usage:  python scripts/experiments/moments_overlap.py <tag>
Everything comes from the dense model, as in the paper: calibration seed 0, half-sets of 64 x 512 tokens packed
exactly as kgcal.run_grid packs them, and the paper's widths. For every block:
1. Second moments S_X = E[z z^T] of z = [h, 1] at the down_proj input, on the half-sets C4a, C4b, KG-Da, KG-Db,
   KG-D~a, KG-Sa, KG-QAa, KG-QA~a, KG-O2a and WIKIa.
2. Moment distance d(X, Y) = ||S_X - S_Y||_F / mean(||S_X||_F, ||S_Y||_F), and the scaled distance, the same on
   the unit block after dividing entry (i, j) by r_i r_j, with r the root mean square of each unit over the two texts
   (a few units with very large activations dominate the raw norm; the scaled one weighs every unit alike). Pairs: the
   paired changes (truth: KG-D vs
   KG-D~ and KG-QA vs KG-QA~; subject: KG-D vs KG-S) and, for scale, resampling the facts (KG-Da vs KG-Db),
   resampling C4 (C4a vs C4b), format (KG-D vs KG-QA), relations (KG-D vs KG-O2), text (KG-D vs WIKI), source
   (KG-D vs C4).
3. Where a paired difference sits (prediction P3): the moments of a paired text are also accumulated over the tokens
   that differ from its partner line (the object, or the subject for KG-S). The two texts are compared within each
   class of tokens, as per-token average moments: over the changed tokens, and over all other tokens, which can differ
   only by attending to changed tokens earlier in the sequence (the two classes have different token counts in the two
   texts, so a split of the summed difference would mix in that count).
4. Kept units: kgcal's activation-norm selection on MIX(X) = (C4a + Xa) / 2 at 70% and 50% of units, and the
   overlap |S_X and S_Y| / M over compressed blocks, for the same pairs (MIX(KG-Da) vs MIX(KG-Db) and C4a vs C4b as
   the C4 half are the resampling floors), and C4/C4's selection vs MIX(KG-D).
5. Repair: at every fourth block (those of the Lemma 1 checks), the ridge consumer of kgcal on MIX(X) for the units
   kept by C4 selection, and ||A_X - A_Y||_F / ||A_X||_F for the same pairs.
Writes results/mechanism/<tag>_moments.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from kgcal import paths as P            # noqa: E402
from kgcal import compress as C        # noqa: E402
from kgcal import data as D            # noqa: E402
from kgcal import model_ops as MO      # noqa: E402

MODELS = {"qwen15b": "Qwen/Qwen2.5-1.5B", "olmo1b": "allenai/OLMo-2-0425-1B",
          "qwen7b": "Qwen/Qwen2.5-7B", "olmo7b": "allenai/OLMo-2-1124-7B"}
GROUP = {"qwen15b": 4, "olmo1b": 4, "qwen7b": 1, "olmo7b": 1}
H = 64                                                     # sequences per half-set, as in the paper
tag = sys.argv[1]
t0 = time.time()
log = lambda *a: print(time.strftime("%H:%M:%S"), f"[{tag}]", *a, flush=True)
OUT = P.MECHANISM
OUT.mkdir(parents=True, exist_ok=True)

lines = {k: [json.loads(l) for l in open(P.calib_file(k))] for k in ("KG-D", "KG-D~", "KG-S", "KG-QA", "KG-QA~")}
lines |= {k: [json.loads(l) for l in open(P.CALIB / f"{k}.jsonl")] for k in ("KG-O2", "WIKI")}


def changed_span(x: str, y: str) -> tuple[int, int]:
    """Characters of x that differ from its paired line y: [common prefix, len(x) - common suffix)."""
    p = 0
    while p < min(len(x), len(y)) and x[p] == y[p]:
        p += 1
    q = 0
    while q < min(len(x), len(y)) - p and x[-1 - q] == y[-1 - q]:
        q += 1
    return p, len(x) - q


def pack_labelled(tok, src: list[str], partner: list[str] | None, n_seq: int):
    """kgcal.data.pack, plus a mask of the tokens that overlap the characters that differ from the partner line."""
    ids, lab = [], []
    for i, t in enumerate(src):
        enc = tok(t + "\n", add_special_tokens=False, return_offsets_mapping=partner is not None)
        ids.extend(enc["input_ids"])
        if partner is not None:
            a, b = changed_span(t, partner[i])
            lab.extend([(s < b and e > a) for s, e in enc["offset_mapping"]])
        if len(ids) >= n_seq * 512:
            break
    x = torch.tensor(ids[: n_seq * 512]).view(n_seq, 512)
    m = torch.tensor(lab[: n_seq * 512]).view(n_seq, 512) if partner is not None else None
    return x, m


model, tok = MO.load(MODELS[tag])
dev = next(model.parameters()).device
L, Nu, d = model.config.num_hidden_layers, model.config.intermediate_size, model.config.hidden_size
dense = MO.dense_copy(model)
c4 = D.c4_lines(3000)
c4_ids = D.pack(tok, c4[:2500], 2 * H)
HS = {"C4a": (c4_ids[:H], {}), "C4b": (c4_ids[H:], {})}
# paired texts: masks of the tokens that differ from the partner (object for truth, subject for KG-S)
for name, src, masks in (("KG-D", "KG-D", {"obj": "KG-D~", "subj": "KG-S"}), ("KG-D~", "KG-D~", {"obj": "KG-D"}),
                         ("KG-S", "KG-S", {"subj": "KG-D"}), ("KG-QA", "KG-QA", {"obj": "KG-QA~"}),
                         ("KG-QA~", "KG-QA~", {"obj": "KG-QA"})):
    ref = D.pack(tok, lines[src], 2 * H)
    ms = {}
    for mk, partner in masks.items():
        x, m = pack_labelled(tok, lines[src], lines[partner], 2 * H)
        assert torch.equal(x, ref), f"labelled packing differs from kgcal.data.pack for {src}"
        ms[mk] = m[:H]
    HS[name + "a"] = (ref[:H], ms)
    if name == "KG-D":
        HS["KG-Db"] = (ref[H:], {})
for k in ("KG-O2", "WIKI"):
    HS[k + "a"] = (D.pack(tok, lines[k], 2 * H)[:H], {})
share = {f"{k}:{mk}": float(m.float().mean()) for k, (_, ms) in HS.items() for mk, m in ms.items()}
log("half-sets", list(HS), "| changed-token shares", {k: round(v, 3) for k, v in share.items()})

dims = json.loads((P.GRID / f"{tag}_gride.json").read_text())["dims"]
WID = {f: MO.widths(dims, MO.alpha_for_fraction(dims, Nu, f), Nu) for f in (0.7, 0.5)}


@torch.no_grad()
def moments(ids, masks, layers, batch=8):
    """Sum of z z^T over all tokens and over each masked token set, per layer (float32 on the GPU, then CPU)."""
    acc = {(l, mk): torch.zeros(Nu + 1, Nu + 1, device=dev) for l in layers for mk in ["all"] + list(masks)}
    last = max(layers)
    cur = {}
    hooks = []
    for l in layers:
        def hk(mod, inp, out, l=l):
            h = inp[0].reshape(-1, Nu).float()
            z = torch.cat([h, torch.ones(h.shape[0], 1, device=dev)], 1)
            acc[(l, "all")].addmm_(z.T, z)
            for mk, m in cur.items():
                zm = z[m]
                acc[(l, mk)].addmm_(zm.T, zm)
            if l == last:
                raise MO.StopForward
        hooks.append(model.model.layers[l].mlp.down_proj.register_forward_hook(hk))
    try:
        for b in range(0, ids.shape[0], batch):
            cur.clear()
            cur.update({mk: m[b:b + batch].reshape(-1).to(dev) for mk, m in masks.items()})
            try:
                model(input_ids=ids[b:b + batch].to(dev))
            except MO.StopForward:
                pass
    finally:
        for h_ in hooks:
            h_.remove()
    n = float(ids.numel())
    return {k: (v / n).cpu() for k, v in acc.items()}


def fro(a):
    return float(torch.linalg.matrix_norm(a.to(dev)))


def dist(a, b):
    return fro(a - b) / ((fro(a) + fro(b)) / 2)


def scaler(a, b):
    """1 / (r_i r_j) on the unit block, r = RMS of each unit over the two texts (units never active: 1)."""
    r = torch.sqrt(((torch.diag(a) + torch.diag(b)) / 2)[:Nu].to(dev).clamp(min=0))
    r[r == 0] = 1
    return 1 / torch.outer(r, r)


def ndist(a, b, sc):
    A, B = a[:Nu, :Nu].to(dev) * sc, b[:Nu, :Nu].to(dev) * sc
    return float(torch.linalg.matrix_norm(A - B) / ((torch.linalg.matrix_norm(A) + torch.linalg.matrix_norm(B)) / 2))


PAIRS = {"truth_decl": ("KG-Da", "KG-D~a"), "truth_qa": ("KG-QAa", "KG-QA~a"), "subject": ("KG-Da", "KG-Sa"),
         "resample_facts": ("KG-Da", "KG-Db"), "resample_c4": ("C4a", "C4b"), "format": ("KG-Da", "KG-QAa"),
         "relations": ("KG-Da", "KG-O2a"), "text": ("KG-Da", "WIKIa"), "source": ("KG-Da", "C4a")}
SPLIT = {"truth_decl": "obj", "truth_qa": "obj", "subject": "subj"}
rows = []
for g0 in range(0, L, GROUP[tag]):
    layers = list(range(g0, min(L, g0 + GROUP[tag])))
    M = {}
    for k, (ids, ms) in HS.items():
        for (l, mk), v in moments(ids, ms, layers).items():
            M[(k, l, mk)] = v
    for l in layers:
        Wd = dense[l]["down_proj"].float()
        r = {"layer": l, "dist": {}, "ndist": {}, "split": {}, "diag_dist": {}, "overlap": {}, "repair": {}}
        for p, (x, y) in PAIRS.items():
            Sx, Sy = M[(x, l, "all")], M[(y, l, "all")]
            r["dist"][p] = dist(Sx, Sy)
            sc = scaler(Sx, Sy)
            r["ndist"][p] = ndist(Sx, Sy, sc)
            dx, dy = torch.sqrt(torch.diag(Sx)[:Nu].clamp(min=0)), torch.sqrt(torch.diag(Sy)[:Nu].clamp(min=0))
            r["diag_dist"][p] = float((dx - dy).norm() / dx.norm())
            if p in SPLIT:
                mk = SPLIT[p]
                fx, fy = share[f"{x}:{mk}"], share[f"{y}:{mk}"]
                cx, cy = M[(x, l, mk)] / fx, M[(y, l, mk)] / fy                       # per changed token
                ux, uy = (Sx - M[(x, l, mk)]) / (1 - fx), (Sy - M[(y, l, mk)]) / (1 - fy)   # per other token
                r["split"][p] = {"changed_tok": dist(cx, cy), "other_tok": dist(ux, uy),
                                 "changed_tok_scaled": ndist(cx, cy, scaler(cx, cy)),
                                 "other_tok_scaled": ndist(ux, uy, scaler(ux, uy)),
                                 "tokens_changed": (fx + fy) / 2}
        # selection sources as (first half-set, second half-set): the moments are their average, as in run_grid
        SRC = {"C4": ("C4a", "C4b"), "MIX:KG-D": ("C4a", "KG-Da"), "MIX:KG-Db": ("C4a", "KG-Db"),
               "MIX:KG-D~": ("C4a", "KG-D~a"), "MIX:KG-S": ("C4a", "KG-Sa"), "MIX:KG-QA": ("C4a", "KG-QAa"),
               "MIX:KG-QA~": ("C4a", "KG-QA~a"), "MIX:KG-O2": ("C4a", "KG-O2a"), "MIX:WIKI": ("C4a", "WIKIa"),
               "MIX:KG-D|C4b": ("C4b", "KG-Da")}
        dg = {k: torch.diag(M[(k, l, "all")]) for k in HS}
        wnorm = Wd.norm(dim=0)

        def select(src, m_):
            """kgcal.compress.select('actnorm') from the diagonal of the averaged moments (identical arithmetic)."""
            a, b = SRC[src]
            score = wnorm * torch.sqrt(((dg[a] + dg[b]) / 2)[:Nu].float().clamp(min=0))
            return torch.argsort(score, descending=True)[:m_].sort().values

        full = lambda src: (M[(SRC[src][0], l, "all")] + M[(SRC[src][1], l, "all")]) / 2
        OV = {"truth_decl": ("MIX:KG-D", "MIX:KG-D~"), "truth_qa": ("MIX:KG-QA", "MIX:KG-QA~"),
              "subject": ("MIX:KG-D", "MIX:KG-S"), "resample_facts": ("MIX:KG-D", "MIX:KG-Db"),
              "resample_c4": ("MIX:KG-D", "MIX:KG-D|C4b"), "format": ("MIX:KG-D", "MIX:KG-QA"),
              "relations": ("MIX:KG-D", "MIX:KG-O2"), "text": ("MIX:KG-D", "MIX:WIKI"), "source": ("MIX:KG-D", "C4")}
        for f in (0.7, 0.5):
            m_ = WID[f][l]
            if m_ >= Nu:
                continue
            S = {k: set(select(k, m_).tolist()) for k in SRC}
            r["overlap"][str(f)] = {p: len(S[a] & S[b]) / m_ for p, (a, b) in OV.items()}
            if l % 4 == 2:
                Sc = C.select("actnorm", Wd, m_, full("C4"), dense[l])
                assert torch.equal(Sc, select("C4", m_)), "diagonal selection differs from kgcal.compress.select"
                A = {k: C.ridge(Wd, Sc, full(k)) for k in ("C4", "MIX:KG-D", "MIX:KG-Db", "MIX:KG-D~", "MIX:KG-S",
                                                           "MIX:KG-QA", "MIX:KG-QA~", "MIX:KG-D|C4b")}
                rel = lambda a, b: float((A[a] - A[b]).norm() / A[a].norm())
                r["repair"][str(f)] = {p: rel(a, b) for p, (a, b) in OV.items() if a in A and b in A}
        rows.append(r)
        log(f"layer {l}: scaled dist truth {r['ndist']['truth_decl']:.4f} facts {r['ndist']['resample_facts']:.4f} "
            f"format {r['ndist']['format']:.4f} source {r['ndist']['source']:.4f} | overlap 70% truth "
            f"{r['overlap'].get('0.7', {}).get('truth_decl', float('nan')):.3f} facts "
            f"{r['overlap'].get('0.7', {}).get('resample_facts', float('nan')):.3f} ({time.time() - t0:.0f}s)")
    del M
    torch.cuda.empty_cache()


def mean_over(key, p, f=None):
    v = [r[key][p] if f is None else r[key][f][p] for r in rows if (f is None or f in r[key]) and
         (p in (r[key] if f is None else r[key][f]))]
    return float(np.mean(v)) if v else None


summary = {"dist": {p: mean_over("dist", p) for p in PAIRS}, "ndist": {p: mean_over("ndist", p) for p in PAIRS},
           "diag_dist": {p: mean_over("diag_dist", p) for p in PAIRS},
           "overlap": {f: {p: mean_over("overlap", p, f) for p in PAIRS} for f in ("0.7", "0.5")},
           "repair": {f: {p: mean_over("repair", p, f) for p in PAIRS} for f in ("0.7", "0.5")},
           "split": {p: {k: float(np.mean([r["split"][p][k] for r in rows]))
                         for k in ("changed_tok", "other_tok", "changed_tok_scaled", "other_tok_scaled", "tokens_changed")}
                     for p in SPLIT}}
(OUT / f"{tag}_moments.json").write_text(json.dumps({"tag": tag, "model": MODELS[tag], "token_share": share,
                                                     "summary": summary, "layers": rows}, indent=1))
log("summary", json.dumps(summary)[:600])
log(f"all done in {time.time() - t0:.0f}s")
print("JOB OK", tag, flush=True)
