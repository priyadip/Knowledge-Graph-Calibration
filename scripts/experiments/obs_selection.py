"""Second structured method: structured OBS selection with ridge repair.

Usage:  python scripts/experiments/obs_selection.py <tag>
- Selection (as in ZipLM / SlimGPT, for the feed-forward units read by down_proj): the N - m units to remove are
  dropped in 8 equal chunks; before each chunk, H_K is the selection-moment matrix E[h_K h_K^T] of the units still
  kept, damped by 1% of its mean diagonal, and the units with the smallest ||W[:, j]||^2 / [H_K^{-1}]_jj (the OBS
  increase in output error when unit j is removed and the others are optimally compensated) are dropped.
- Repair: the paper's ridge (kgcal.compress.ridge, lambda 1e-4) on the repair moments; "none" keeps the weights.
- Conditions (sample 0, 70% and 50%, the paper's widths): C4|C4, C4|MIX:KG-D, MIX:KG-D|MIX:KG-D,
  MIX:KG-D~|MIX:KG-D~ (paired file of data/calib_paired), MIX:KG-D|none. Question and cloze retention, perplexity;
  ARC-Easy and HellaSwag on the 7B models for C4|C4 and MIX:KG-D|MIX:KG-D.
Outputs: results/obs/<tag>_obs.jsonl and <tag>_obs_bits.json.
"""
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from kgcal import paths as P            # noqa: E402
from kgcal import compress as C        # noqa: E402
from kgcal import data as D            # noqa: E402
from kgcal import evaluate as E        # noqa: E402
from kgcal import model_ops as MO      # noqa: E402
from kgcal import run_grid as R        # noqa: E402

MODELS = {"qwen15b": "Qwen/Qwen2.5-1.5B", "olmo1b": "allenai/OLMo-2-0425-1B",
          "qwen7b": "Qwen/Qwen2.5-7B", "olmo7b": "allenai/OLMo-2-1124-7B"}
BATCH = {"qwen15b": 96, "olmo1b": 96, "qwen7b": 64, "olmo7b": 64}
GROUP = {"qwen15b": 4, "olmo1b": 4, "qwen7b": 1, "olmo7b": 1}
SEVEN = {"qwen7b", "olmo7b"}
CONDS = [("C4|C4", "C4", "C4"), ("C4|MIX:KG-D", "C4", "MIX:KG-D"), ("MIX:KG-D|MIX:KG-D", "MIX:KG-D", "MIX:KG-D"),
         ("MIX:KG-D~|MIX:KG-D~", "MIX:KG-D~", "MIX:KG-D~"), ("MIX:KG-D|none", "MIX:KG-D", None)]
ZS = {"C4|C4", "MIX:KG-D|MIX:KG-D"}
H, CHUNKS, DAMP = 64, 8, 0.01


def obs_select(Wd: torch.Tensor, M2: torch.Tensor, m: int) -> torch.Tensor:
    """Kept unit indices (sorted) after chunked structured OBS removal down to m units."""
    N = Wd.shape[1]
    dev = "cuda"
    Hm = M2[:N, :N].to(dev).float()
    w2 = (Wd.to(dev).float() ** 2).sum(0)
    keep = torch.arange(N, device=dev)
    remove = N - m
    for c in range(CHUNKS):
        k = remove * (c + 1) // CHUNKS - remove * c // CHUNKS
        if k == 0:
            continue
        Hk = Hm[keep][:, keep]
        Hk = Hk + DAMP * Hk.diagonal().mean() * torch.eye(len(keep), device=dev)
        dinv = torch.linalg.inv(Hk).diagonal()
        score = w2[keep] / dinv
        drop = torch.argsort(score)[:k]
        mask = torch.ones(len(keep), dtype=torch.bool, device=dev)
        mask[drop] = False
        keep = keep[mask]
        del Hk, dinv
    assert len(keep) == m
    return keep.sort().values.cpu()


def main():
    tag = sys.argv[1]
    t0 = time.time()
    log = lambda *a: print(time.strftime("%H:%M:%S"), f"[{tag}_obs]", *a, flush=True)
    out_dir = P.OBS
    out_dir.mkdir(parents=True, exist_ok=True)
    res_path, bits_path = out_dir / f"{tag}_obs.jsonl", out_dir / f"{tag}_obs_bits.json"
    rows = D.load_popqa()
    F = D.facts_by_split(rows, D.make_split(rows))
    srcs = {"KG-D": [json.loads(l) for l in open(D.CALIB / "KG-D.jsonl")],
            "KG-D~": [json.loads(l) for l in open(P.calib_file("KG-D~"))]}
    c4 = D.c4_lines(3000)
    model, tok = MO.load(MODELS[tag])
    dense = MO.dense_copy(model)
    L, N, d = model.config.num_hidden_layers, model.config.intermediate_size, model.config.hidden_size
    total_params = sum(p.numel() for p in model.parameters())
    c4_ids = D.pack(tok, c4[:2500], 2 * H)
    half = {"C4a": c4_ids[:H], "C4b": c4_ids[H:]}
    for k, lines in srcs.items():
        ids = D.pack(tok, lines, 2 * H)
        half[k + "a"], half[k + "b"] = ids[:H], ids[H:]
    need = sorted({"C4a", "C4b", "KG-Da", "KG-D~a"})
    dims = json.loads((P.GRID / f"{tag}_gride.json").read_text())["dims"]
    budgets = []
    for f in (0.7, 0.5):
        a = MO.alpha_for_fraction(dims, N, f)
        w = MO.widths(dims, a, N)
        removed = sum((N - m) * 3 * d for m in w)
        budgets.append({"target": f, "widths": w, "param_keep": (total_params - removed) / total_params})
    store = {}
    for g0 in range(0, L, GROUP[tag]):
        layers = list(range(g0, min(L, g0 + GROUP[tag])))
        mom = {h: MO.collect_moments(model, half[h], layers) for h in need}
        for b in budgets:
            for i in layers:
                Wd = dense[i]["down_proj"].float()
                sel = {}
                for cname, ssrc, rsrc in CONDS:
                    if ssrc not in sel:
                        sel[ssrc] = obs_select(Wd, R.moments_of(ssrc, mom, i), b["widths"][i])
                    S = sel[ssrc]
                    A = None if rsrc is None else C.ridge(Wd, S, R.moments_of(rsrc, mom, i), lam_rel=1e-4).to(torch.bfloat16)
                    store.setdefault((b["target"], cname), {})[i] = (S, A, None)
        del mom
        torch.cuda.empty_cache()
        log(f"layers {layers[0]}-{layers[-1]} done ({time.time() - t0:.0f}s)")

    ex = E.shots(F["cal"])
    pp = E.ppl_sets(tok, c4[2600:3000])
    zs = E.zeroshot_sets() if tag in SEVEN else None
    bits = json.loads(bits_path.read_text()) if bits_path.exists() else {}
    for b in budgets:
        for cname, _, _ in CONDS:
            key = f"{b['target']}|{cname}"
            if key in bits:
                continue
            t = time.time()
            for i in range(L):
                MO.apply(model, dense, i, *store[(b["target"], cname)][i])
            tq = E.recall(model, tok, F["test"], ex, "question", BATCH[tag])
            tc = E.recall(model, tok, F["test"], ex, "cloze", BATCH[tag])
            rec = {"tag": tag, "cond": cname, "target": b["target"], "method": "obs", "param_keep": b["param_keep"],
                   "test_q_acc": sum(tq) / len(tq), "test_c_acc": sum(tc) / len(tc),
                   "ppl_wikitext2": E.perplexity(model, pp["wikitext2"]), "ppl_c4": E.perplexity(model, pp["c4"])}
            if zs is not None and cname in ZS:
                rec["arc_easy"] = E.multiple_choice(model, tok, zs["arc_easy"], length_norm=False)
                rec["hellaswag"] = E.multiple_choice(model, tok, zs["hellaswag"], length_norm=True)
            rec["secs"] = round(time.time() - t, 1)
            bits[key] = {"test_q": "".join("1" if x else "0" for x in tq), "test_c": "".join("1" if x else "0" for x in tc)}
            with open(res_path, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            bits_path.write_text(json.dumps(bits))
            log(f"{key:26s} test_q={rec['test_q_acc']:.4f} test_c={rec['test_c_acc']:.4f} "
                f"ppl_wt={rec['ppl_wikitext2']:.2f} {rec.get('hellaswag', '')} ({rec['secs']}s)")
            MO.restore(model, dense)
    log(f"all done in {time.time() - t0:.0f}s")
    print("JOB OK", tag, flush=True)


if __name__ == "__main__":
    main()
