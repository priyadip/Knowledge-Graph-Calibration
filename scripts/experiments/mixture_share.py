"""Mixture sweep: share of KG statements in a fixed calibration budget.

Usage:  python scripts/experiments/mixture_share.py <tag> [cseed] [--extra]
- cseed 1, 2: another calibration sample, drawn exactly as kgcal.run_grid does (KG-D shuffled with
  random.Random(cseed), C4 from the disjoint stream); R0|R0 and R64|R64 must reproduce the paper's sample-cseed runs.
- Calibration: 128 sequences of 512 tokens. n KG-D sequences and 128 - n C4 sequences, n in {13, 32, 96}
  (10.2%, 25%, 75%), taken as prefixes of the paper's packed pools (C4 = C4a then C4b, KG-D = KG-Da then KG-Db).
  Moments are token weighted: sum over pieces of (sequences x piece mean) / 128.
- Channels for every share: both (selection and repair on the mixture), repair only (selection on C4) and selection
  only without repair. Replay checks: shares 0 and 50 are built from the full half-sets with the original moment
  collector, so R0|R0 must reproduce the paper's C4|C4 and R64|R64 its MIX:KG-D|MIX:KG-D bit for bit.
- 7B models also get ARC-Easy and HellaSwag for every condition.
--extra adds selection on the 50% mixture without repair. Everything else (widths, selection, ridge 1e-4, prompts,
batch, scoring) is kgcal as used in the paper. Outputs: results/mixture/<tag>_mix.jsonl and <tag>_mix_bits.json.
"""
import json
import random
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

MODELS = {"qwen15b": "Qwen/Qwen2.5-1.5B", "olmo1b": "allenai/OLMo-2-0425-1B",
          "qwen7b": "Qwen/Qwen2.5-7B", "olmo7b": "allenai/OLMo-2-1124-7B"}
BATCH = {"qwen15b": 96, "olmo1b": 96, "qwen7b": 64, "olmo7b": 64}
GROUP = {"qwen15b": 4, "olmo1b": 4, "qwen7b": 1, "olmo7b": 1}
SEVEN_B = {"qwen7b", "olmo7b"}
H, TOTAL = 64, 128
SHARES = [13, 32, 96]


def pieces(n: int) -> list[tuple[str, int, int]]:
    """(half-set, prefix length, weight) making up a mixture of n KG-D and 128 - n C4 sequences."""
    out = []
    c = TOTAL - n
    for name, k in (("C4a", min(c, H)), ("C4b", max(c - H, 0)), ("KG-Da", min(n, H)), ("KG-Db", max(n - H, 0))):
        if k:
            out.append((name, k, k))
    return out


def conditions(extra: bool = False) -> list[tuple]:
    """(name, selection source or None, repair source or None) with sources R<n>."""
    if extra:                                              # --extra: 50% selection without repair
        return [("R0|R0", "R0", "R0"), ("R64|none", "R64", None)]
    conds = [("R0|R0", "R0", "R0"), ("R64|R64", "R64", "R64")]
    for n in SHARES:
        conds += [(f"R{n}|R{n}", f"R{n}", f"R{n}"), (f"R0|R{n}", "R0", f"R{n}"), (f"R{n}|none", f"R{n}", None)]
    return conds


def main():
    tag = sys.argv[1]
    extra = "--extra" in sys.argv
    argv = [a for a in sys.argv if a != "--extra"]
    cseed = int(argv[2]) if len(argv) > 2 else 0
    sfx = f"_cs{cseed}" if cseed else ""
    t0 = time.time()
    log = lambda *a: print(time.strftime("%H:%M:%S"), f"[{tag}_mix{sfx}]", *a, flush=True)
    out_dir = P.MIXTURE
    out_dir.mkdir(parents=True, exist_ok=True)
    res_path, bits_path = out_dir / f"{tag}_mix{sfx}.jsonl", out_dir / f"{tag}_mix{sfx}_bits.json"

    rows = D.load_popqa()
    F = D.facts_by_split(rows, D.make_split(rows))
    kg = [json.loads(l) for l in open(D.CALIB / "KG-D.jsonl")]
    if cseed:
        random.Random(cseed).shuffle(kg)                  # KG-D is the first source run_grid shuffles
    c4 = D.c4_lines(3000)
    model, tok = MO.load(MODELS[tag])
    dense = MO.dense_copy(model)
    L, N, d = model.config.num_hidden_layers, model.config.intermediate_size, model.config.hidden_size
    total_params = sum(p.numel() for p in model.parameters())
    c4_cal = D.c4_lines(2500, skip=3000 + 2500 * (cseed - 1)) if cseed else c4[:2500]
    c4_ids, kg_ids = D.pack(tok, c4_cal, 2 * H), D.pack(tok, kg, 2 * H)
    half = {"C4a": c4_ids[:H], "C4b": c4_ids[H:], "KG-Da": kg_ids[:H], "KG-Db": kg_ids[H:]}
    conds = conditions(extra)
    srcs = sorted({s for c in conds for s in c[1:] if s})
    need = sorted({(p, k) for s in srcs for p, k, _ in pieces(int(s[1:]))})
    log("pieces:", need)

    dims = json.loads((P.GRID / f"{tag}_gride.json").read_text())["dims"]
    budgets = []
    for f in (0.7, 0.5):
        a = MO.alpha_for_fraction(dims, N, f)
        w = MO.widths(dims, a, N)
        removed = sum((N - m) * 3 * d for m in w)
        budgets.append({"target": f, "alpha": a, "widths": w, "ffn_keep": sum(w) / (L * N),
                        "param_keep": (total_params - removed) / total_params})

    store: dict[tuple, dict[int, tuple]] = {}
    for g0 in range(0, L, GROUP[tag]):
        layers = list(range(g0, min(L, g0 + GROUP[tag])))
        # full half-sets with the paper's collector (replay); prefixes collected on their own sequences
        mom = {(p, k): MO.collect_moments(model, half[p] if k == H else half[p][:k], layers) for p, k in need}

        def moments(src, i):
            ps = pieces(int(src[1:]))
            if src == "R0":                                   # exactly run_grid.moments_of("C4")
                return (mom[("C4a", H)][i] + mom[("C4b", H)][i]) / 2
            if src == "R64":                                  # exactly run_grid.moments_of("MIX:KG-D")
                return (mom[("C4a", H)][i] + mom[("KG-Da", H)][i]) / 2
            return sum(wt * mom[(p, k)][i] for p, k, wt in ps) / TOTAL

        for b in budgets:
            for i in layers:
                Wd = dense[i]["down_proj"].float()
                m = b["widths"][i]
                sel = {}
                for cname, ssrc, rsrc in conds:
                    if ssrc not in sel:
                        sel[ssrc] = C.select("actnorm", Wd, m, moments(ssrc, i), dense[i], seed=cseed)
                    S = sel[ssrc]
                    A = None if rsrc is None else C.ridge(Wd, S, moments(rsrc, i), lam_rel=1e-4).to(torch.bfloat16)
                    store.setdefault((b["target"], cname), {})[i] = (S, A, None)
        del mom
        torch.cuda.empty_cache()
        log(f"layers {layers[0]}-{layers[-1]} done ({time.time() - t0:.0f}s)")

    ex = E.shots(F["cal"])
    pp = E.ppl_sets(tok, c4[2600:3000])
    zs = E.zeroshot_sets() if tag in SEVEN_B else None
    bits = json.loads(bits_path.read_text()) if bits_path.exists() else {}
    bs = BATCH[tag]

    def evaluate(name, b):
        t = time.time()
        rec = {"tag": tag, "cond": name, "target": b["target"] if b else 1.0,
               "ffn_keep": b["ffn_keep"] if b else 1.0, "param_keep": b["param_keep"] if b else 1.0}
        vq = E.recall(model, tok, F["val"], ex, "question", bs)
        tq = E.recall(model, tok, F["test"], ex, "question", bs)
        tc = E.recall(model, tok, F["test"], ex, "cloze", bs)
        rec.update({"val_q_acc": sum(vq) / len(vq), "test_q_acc": sum(tq) / len(tq), "test_c_acc": sum(tc) / len(tc),
                    "ppl_wikitext2": E.perplexity(model, pp["wikitext2"]), "ppl_c4": E.perplexity(model, pp["c4"])})
        if zs is not None:
            rec["arc_easy"] = E.multiple_choice(model, tok, zs["arc_easy"], length_norm=False)
            rec["hellaswag"] = E.multiple_choice(model, tok, zs["hellaswag"], length_norm=True)
        rec["secs"] = round(time.time() - t, 1)
        key = f"{rec['target']}|{name}"
        bits[key] = {k: "".join("1" if x else "0" for x in v) for k, v in (("val_q", vq), ("test_q", tq), ("test_c", tc))}
        with open(res_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        bits_path.write_text(json.dumps(bits))
        log(f"{key:20s} test_q={rec['test_q_acc']:.4f} test_c={rec['test_c_acc']:.4f} "
            f"ppl_wt={rec['ppl_wikitext2']:.2f} {rec.get('hellaswag', '')} ({rec['secs']}s)")

    if "1.0|dense" not in bits:
        evaluate("dense", None)
    for b in budgets:
        for cname, _, _ in conds:
            if f"{b['target']}|{cname}" in bits:
                continue
            for i in range(L):
                S, A, bias = store[(b["target"], cname)][i]
                MO.apply(model, dense, i, S, A, bias)
            evaluate(cname, b)
            MO.restore(model, dense)

    ref = json.loads((P.MERGED / f"{tag}{sfx}_bits.json").read_text())
    pairs = [("1.0|dense", "1.0|dense"), ("0.7|R0|R0", "0.7|C4|C4"), ("0.5|R0|R0", "0.5|C4|C4")]
    pairs += ([("0.7|R64|none", "0.7|MIX:KG-D|none"), ("0.5|R64|none", "0.5|MIX:KG-D|none")] if extra else
              [("0.7|R64|R64", "0.7|MIX:KG-D|MIX:KG-D"), ("0.5|R64|R64", "0.5|MIX:KG-D|MIX:KG-D")])
    for mine, theirs in pairs:
        same = bits[mine]["test_q"] == ref[theirs]["test_q"]
        log(f"replay {mine} vs paper {theirs}: {'identical' if same else 'DIFFERENT'}")
        assert same, mine
    log(f"all done in {time.time() - t0:.0f}s")
    print("JOB OK", tag, flush=True)


if __name__ == "__main__":
    main()
