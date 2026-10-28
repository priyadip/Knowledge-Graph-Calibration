"""One model, several budgets, a grid of calibration conditions, all at identical GRIDE widths.

Usage: python -m kgcal.run_grid --model Qwen/Qwen2.5-1.5B --tag qwen15b --fractions 0.7,0.5 --grid core [--theory]
Writes results/grid/<tag>.jsonl (one line per configuration), <tag>_bits.json (per-fact correctness),
<tag>_gride.json (layer dimensions), <tag>_theory.jsonl (Lemma 1 terms, with --theory).
"""
from __future__ import annotations

import argparse
import json
import random
import time

import torch

from . import compress as C
from . import data as D
from . import evaluate as E
from . import model_ops as MO

KG_VARIANTS = ["KG-D~", "KG-QA", "KG-QA~", "KG-S", "WIKI", "KG-O", "KG-A"]


def grid(name: str) -> list[tuple]:
    """(condition name, selection rule, selection source, repair, repair source)."""
    core = [("none|C4", "actnorm", "C4", "none", None),
            ("bias|C4", "actnorm", "C4", "bias", "C4"),
            ("C4|C4", "actnorm", "C4", "ridge", "C4"),
            ("C4|MIX:KG-D", "actnorm", "C4", "ridge", "MIX:KG-D"),
            ("MIX:KG-D|none", "actnorm", "MIX:KG-D", "none", None),
            ("MIX:KG-D|C4", "actnorm", "MIX:KG-D", "ridge", "C4"),
            ("MIX:KG-D|MIX:KG-D", "actnorm", "MIX:KG-D", "ridge", "MIX:KG-D")]
    core += [(f"MIX:{x}|MIX:{x}", "actnorm", f"MIX:{x}", "ridge", f"MIX:{x}") for x in KG_VARIANTS]
    ext = [("mag|none", "magnitude", None, "none", None),
           ("mag|C4", "magnitude", None, "ridge", "C4"),
           ("mag|MIX:KG-D", "magnitude", None, "ridge", "MIX:KG-D"),
           ("rand|C4", "random", None, "ridge", "C4"),
           ("rand|MIX:KG-D", "random", None, "ridge", "MIX:KG-D"),
           ("KG-D|KG-D", "actnorm", "KG-D", "ridge", "KG-D"),
           ("C4|KG-D", "actnorm", "C4", "ridge", "KG-D"),
           ("MIX:KG-D|bias", "actnorm", "MIX:KG-D", "bias", "MIX:KG-D")]
    anchor_clean = [("MIX:KG-Ac|MIX:KG-Ac", "actnorm", "MIX:KG-Ac", "ridge", "MIX:KG-Ac")]
    followup = anchor_clean + [("MIX:KG-O2|MIX:KG-O2", "actnorm", "MIX:KG-O2", "ridge", "MIX:KG-O2")]
    # corrected sources for the 7B runs: KG-O2 (fixed verbalizer) and KG-Ac (clean anchored) replace the v1 sets
    core7b = [c for c in core if c[0] not in ("MIX:KG-O|MIX:KG-O", "MIX:KG-A|MIX:KG-A")] + followup
    lam = [c for c in core if c[0] in ("none|C4", "C4|C4", "C4|MIX:KG-D", "MIX:KG-D|MIX:KG-D")]
    c4only = [c for c in core if c[0] in ("none|C4", "C4|C4")]
    # the consumers of Figure 2, for layer-level measurement only (with --theory_only)
    theory = [c for c in core if c[0] in ("C4|C4", "C4|MIX:KG-D", "MIX:KG-D|MIX:KG-D", "MIX:KG-D~|MIX:KG-D~",
                                          "MIX:KG-QA|MIX:KG-QA", "MIX:KG-QA~|MIX:KG-QA~")] \
        + [c for c in ext if c[0] == "C4|KG-D"]
    key = [c for c in core if c[0] in ("none|C4", "C4|C4", "MIX:KG-D|MIX:KG-D", "MIX:KG-D~|MIX:KG-D~",
                                       "MIX:KG-QA|MIX:KG-QA", "MIX:KG-QA~|MIX:KG-QA~")]
    return {"core": core, "ext": ext, "key": key, "all": core + ext, "anchor_clean": anchor_clean,
            "followup": followup, "core7b": core7b, "lam": lam, "c4only": c4only, "theory": theory}[name]


def halfsets_for(conds: list[tuple]) -> set[str]:
    need = set()
    for _, rule, ssrc, rep, rsrc in conds:
        for src in (ssrc if rule == "actnorm" else None, rsrc):
            if src is None:
                continue
            if src == "C4":
                need |= {"C4a", "C4b"}
            elif src.startswith("MIX:"):
                need |= {"C4a", src[4:] + "a"}
            else:
                need |= {src + "a", src + "b"}
    return need


def moments_of(src: str, mom: dict, i: int) -> torch.Tensor:
    if src == "C4":
        return (mom["C4a"][i] + mom["C4b"][i]) / 2
    if src.startswith("MIX:"):
        return (mom["C4a"][i] + mom[src[4:] + "a"][i]) / 2
    return (mom[src + "a"][i] + mom[src + "b"][i]) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--fractions", default="0.7,0.5")
    ap.add_argument("--grid", default="core")
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--cseed", type=int, default=0, help="calibration sampling seed")
    ap.add_argument("--theory", action="store_true")
    ap.add_argument("--theory_only", action="store_true", help="stop after the layer statistics (implies --theory)")
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--no_cloze", action="store_true")
    ap.add_argument("--lam", type=float, default=1e-4, help="ridge strength relative to the mean kept diagonal")
    ap.add_argument("--half", type=int, default=64, help="sequences of 512 tokens per calibration half-set")
    ap.add_argument("--zeroshot", default="", help="comma-separated condition names to also score on ARC-Easy and HellaSwag")
    args = ap.parse_args()
    args.theory = args.theory or args.theory_only
    t0 = time.time()
    log = lambda *a: print(time.strftime("%H:%M:%S"), f"[{args.tag}]", *a, flush=True)
    out_dir = D.RESULTS
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.cseed == 0 else f"_cs{args.cseed}"
    res_path, bits_path = out_dir / f"{args.tag}{suffix}.jsonl", out_dir / f"{args.tag}{suffix}_bits.json"

    # ---------------- data
    rows = D.load_popqa()
    split = D.make_split(rows)
    F = D.facts_by_split(rows, split)
    conds = grid(args.grid)
    used = {src.replace("MIX:", "") for c in conds for src in (c[2], c[4]) if src and src != "C4"}
    srcs = {k: [json.loads(l) for l in open(D.calib_path(k))] for k in sorted(used)}
    rng = random.Random(args.cseed)
    for k in srcs:
        if args.cseed:
            rng.shuffle(srcs[k])
    c4 = D.c4_lines(3000)
    c4_cal = c4[:2500]
    if args.cseed:
        c4_cal = D.c4_lines(2500, skip=3000 + 2500 * (args.cseed - 1))

    model, tok = MO.load(args.model)
    dense = MO.dense_copy(model)
    L, N, d = model.config.num_hidden_layers, model.config.intermediate_size, model.config.hidden_size
    total_params = sum(p.numel() for p in model.parameters())

    half = {}
    need = halfsets_for(conds) | {"C4a"}
    H = args.half
    c4_ids = D.pack(tok, c4_cal, 2 * H)
    half["C4a"], half["C4b"] = c4_ids[:H], c4_ids[H:]
    for k, lines in srcs.items():
        if k + "a" in need or k + "b" in need:
            ids = D.pack(tok, lines, 2 * H)
            half[k + "a"], half[k + "b"] = ids[:H], ids[H:]
    if args.theory:
        test_qa = [f"Q: {r['question']} A: {D.ans(r)[0]}" for r in F["test"]]
        test_d = [D.declarative(r["prop"], r["subj"], D.ans(r)[0]) for r in F["test"]]
        def pack_upto(lines, max_seq=128):
            avail = sum(len(tok(t + "\n", add_special_tokens=False)["input_ids"]) for t in lines)
            n = min(max_seq, (avail // 512) // 2 * 2)
            return D.pack(tok, lines, n)
        pq, pc = pack_upto(test_qa), pack_upto(test_d)
        hq, hc = pq.shape[0] // 2, pc.shape[0] // 2
        log(f"theory sets: question {pq.shape[0]} x 512 tokens, cloze {pc.shape[0]} x 512 tokens")
        half.update({"Pq_fit": pq[:hq], "Pq_eval": pq[hq:], "Pc_fit": pc[:hc], "Pc_eval": pc[hc:],
                     "C4h": D.pack(tok, c4[2600:3000], 64)})
        need |= {"Pq_fit", "Pq_eval", "Pc_fit", "Pc_eval", "C4h"}
    need = sorted(h for h in need if h in half)
    log("half-sets:", need)

    # ---------------- GRIDE widths (cached per model)
    gpath = out_dir / f"{args.tag}_gride.json"
    if gpath.exists():
        dims = json.loads(gpath.read_text())["dims"]
    else:
        dims = MO.gride_dims(MO.collect_clouds(model, half["C4a"]))
        gpath.write_text(json.dumps({"model": args.model, "dims": dims}))
    budgets = []
    for f in [float(x) for x in args.fractions.split(",")]:
        a = MO.alpha_for_fraction(dims, N, f)
        w = MO.widths(dims, a, N)
        removed = sum((N - m) * 3 * d for m in w)
        budgets.append({"target": f, "alpha": a, "widths": w, "ffn_keep": sum(w) / (L * N),
                        "param_keep": (total_params - removed) / total_params})
    log("GRIDE d_l:", [round(x, 1) for x in dims])
    log("budgets:", [(b["target"], b["alpha"], round(b["ffn_keep"], 3), round(b["param_keep"], 3)) for b in budgets])

    # ---------------- statistics by layer groups -> kept sets and consumers for every condition
    store: dict[tuple, dict[int, tuple]] = {}
    theory_rows = []
    theory_layers = {i for i in range(L) if i % 4 == 2}
    for g0 in range(0, L, args.group):
        layers = list(range(g0, min(L, g0 + args.group)))
        mom = {h: MO.collect_moments(model, half[h], layers) for h in need}
        for b in budgets:
            for i in layers:
                W = dense[i]["down_proj"].float()
                m = b["widths"][i]
                sel_cache = {}
                for cname, rule, ssrc, rep, rsrc in conds:
                    key = (rule, ssrc)
                    if key not in sel_cache:
                        M2s = moments_of(ssrc, mom, i) if rule == "actnorm" else None
                        sel_cache[key] = C.select(rule, W, m, M2s, dense[i], seed=args.cseed)
                    S = sel_cache[key]
                    if rep == "none":
                        store.setdefault((b["target"], cname), {})[i] = (S, None, None)
                    elif rep == "bias":
                        bias = C.bias_only(W, S, moments_of(rsrc, mom, i))
                        store.setdefault((b["target"], cname), {})[i] = (S, None, bias.float())
                    else:
                        A = C.ridge(W, S, moments_of(rsrc, mom, i), lam_rel=args.lam)
                        store.setdefault((b["target"], cname), {})[i] = (S, A.to(torch.bfloat16), None)
                if args.theory and i in theory_layers and m < N:
                    for (rule, ssrc), S in sel_cache.items():
                        cons = {"none": C.consumer_none(W, S)}
                        for cname, r2, s2, rep, rsrc in conds:
                            if (r2, s2) == (rule, ssrc) and rep == "ridge":
                                cons[f"ridge:{rsrc}"] = store[(b["target"], cname)][i][1].double()
                        row = {"tag": args.tag, "target": b["target"], "layer": i, "sel": f"{rule}:{ssrc}"}
                        for pname, fit, ev in (("question", "Pq_fit", "Pq_eval"), ("cloze", "Pc_fit", "Pc_eval")):
                            row[pname] = C.lemma_terms(mom[fit][i], mom[ev][i], W, S, cons)
                        base = C.error(mom["C4h"][i], W, S, cons["none"])
                        row["web_heldout_ratio"] = {k: C.error(mom["C4h"][i], W, S, A) / base for k, A in cons.items()}
                        theory_rows.append(row)
        del mom
        torch.cuda.empty_cache()
        log(f"layers {layers[0]}-{layers[-1]} done ({time.time() - t0:.0f}s)")
    if args.theory:
        with open(out_dir / f"{args.tag}_theory.jsonl", "w") as f:
            for r in theory_rows:
                f.write(json.dumps(r) + "\n")

    if args.theory_only:
        log(f"all done in {time.time() - t0:.0f}s (theory only)")
        return

    # ---------------- evaluation
    ex = E.shots(F["cal"])
    pp = E.ppl_sets(tok, c4[2600:3000])
    zs_conds = {c for c in args.zeroshot.split(",") if c}
    zs = E.zeroshot_sets() if zs_conds else None
    bits = json.loads(bits_path.read_text()) if bits_path.exists() else {}

    def evaluate(name: str, b: dict | None, spec: tuple | None):
        t = time.time()
        rec = {"tag": args.tag, "model": args.model, "cseed": args.cseed, "cond": name,
               "target": b["target"] if b else 1.0, "alpha": b["alpha"] if b else None,
               "ffn_keep": b["ffn_keep"] if b else 1.0, "param_keep": b["param_keep"] if b else 1.0}
        if spec:
            rec.update({"sel_rule": spec[1], "sel_src": spec[2], "rep": spec[3], "rep_src": spec[4]})
        vq = E.recall(model, tok, F["val"], ex, "question", args.batch)
        tq = E.recall(model, tok, F["test"], ex, "question", args.batch)
        tc = [] if args.no_cloze else E.recall(model, tok, F["test"], ex, "cloze", args.batch)
        rec.update({"val_q_acc": sum(vq) / len(vq), "test_q_acc": sum(tq) / len(tq),
                    "test_c_acc": (sum(tc) / len(tc)) if tc else None,
                    "ppl_wikitext2": E.perplexity(model, pp["wikitext2"]), "ppl_c4": E.perplexity(model, pp["c4"]),
                    "secs": round(time.time() - t, 1)})
        if zs is not None and (name in zs_conds or name == "dense"):
            rec["arc_easy"] = E.multiple_choice(model, tok, zs["arc_easy"], length_norm=False)
            rec["hellaswag"] = E.multiple_choice(model, tok, zs["hellaswag"], length_norm=True)
        key = f"{rec['target']}|{name}"
        bits[key] = {"val_q": "".join("1" if x else "0" for x in vq), "test_q": "".join("1" if x else "0" for x in tq),
                     "test_c": "".join("1" if x else "0" for x in tc)}
        with open(res_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        bits_path.write_text(json.dumps(bits))
        log(f"{key:34s} val_q={rec['val_q_acc']:.4f} test_q={rec['test_q_acc']:.4f} "
            f"test_c={rec['test_c_acc'] if rec['test_c_acc'] is None else round(rec['test_c_acc'], 4)} "
            f"ppl_wt={rec['ppl_wikitext2']:.2f} ppl_c4={rec['ppl_c4']:.2f} ({rec['secs']}s)")

    if "1.0|dense" not in bits:
        evaluate("dense", None, None)
    for b in budgets:
        for spec in conds:
            cname = spec[0]
            if f"{b['target']}|{cname}" in bits:
                continue
            for i in range(L):
                S, A, bias = store[(b["target"], cname)][i]
                MO.apply(model, dense, i, S, A, bias)
            evaluate(cname, b, spec)
            MO.restore(model, dense)
    log(f"all done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
