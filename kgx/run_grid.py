"""kgcal.run_grid for Llama-3.1, Mistral and Phi-4: one model, its budgets, a grid of calibration conditions.

Usage (from the repository root):
  python -m kgx.run_grid --model llama8b --tag llama8b --fractions 0.7,0.5 --grid core7b --group 2 --batch 64
Writes results/families/<tag>[_cs<seed>].jsonl, ..._bits.json, <tag>_gride.json, <tag>_theory.jsonl.

Differences from kgcal.run_grid, and nothing else:
1. Models come from models.REGISTRY at pinned revisions (Phi-4's fused FFN is split).
2. Calibration texts come from data/calib and data/calib_paired, where the counterfactual and made-up-subject texts
   are paired line by line with their true texts. A calibration-sampling seed resamples a variant together with its
   true text (data.paired_shuffle), as the paired reruns of the paper do.
3. BOS families (Llama 3.1, Mistral) get BOS at the start of every calibration sequence, perplexity window
   and zero-shot context. Qwen2.5, OLMo-2 and Phi-4 use no BOS, and for them every sequence is built as in kgcal.
4. --theory_only measures only the layers whose Lemma 1 terms are reported (every fourth block, as in kgcal); every
   statistic comes from the dense model, so the other layers do not affect these terms.
5. --gride_from copies the layer dimensions of another tag (the variant runs of the paper reused the main widths),
   --dims_file sets them from a file (GRIDE-MLE widths of the width check).
6. Each record also stores the generation batch size and the BOS id.
The grid definitions, selection rules, consumers, ridge strength, Lemma 1 terms, width rule, facts, prompts and
metrics are imported from kgcal unchanged.
"""
from __future__ import annotations

import argparse
import json
import time

import torch

from kgcal import compress as C
from kgcal.run_grid import grid, halfsets_for, moments_of  # noqa: F401  (unchanged grid definitions)

from . import data as D
from . import evaluate as E
from . import model_ops as MO
from .models import REGISTRY, load, text_config


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(REGISTRY))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--fractions", default="0.7,0.5")
    ap.add_argument("--grid", default="core")
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--cseed", type=int, default=0, help="calibration sampling seed")
    ap.add_argument("--theory", action="store_true")
    ap.add_argument("--theory_only", action="store_true", help="stop after the layer statistics (implies --theory)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--no_cloze", action="store_true")
    ap.add_argument("--lam", type=float, default=1e-4, help="ridge strength relative to the mean kept diagonal")
    ap.add_argument("--half", type=int, default=64, help="sequences of 512 tokens per calibration half-set")
    ap.add_argument("--zeroshot", default="", help="comma-separated condition names to also score on ARC-Easy and HellaSwag")
    ap.add_argument("--gride_from", default="", help="reuse the layer dimensions of this tag")
    ap.add_argument("--dims_file", default="", help="json file with {'dims': [...]} to use as layer dimensions")
    args = ap.parse_args(argv)
    args.theory = args.theory or args.theory_only
    t0 = time.time()
    log = lambda *a: print(time.strftime("%H:%M:%S"), f"[{args.tag}]", *a, flush=True)
    out_dir = D.RESULTS
    out_dir.mkdir(exist_ok=True)
    suffix = "" if args.cseed == 0 else f"_cs{args.cseed}"
    res_path, bits_path = out_dir / f"{args.tag}{suffix}.jsonl", out_dir / f"{args.tag}{suffix}_bits.json"

    # ---------------- data
    rows = D.load_popqa()
    split = D.make_split(rows)
    F = D.facts_by_split(rows, split)
    conds = grid(args.grid)
    used = {src.replace("MIX:", "") for c in conds for src in (c[2], c[4]) if src and src != "C4"}
    srcs = {k: [json.loads(l) for l in open(D.calib_path(k))] for k in sorted(used)}
    if args.cseed:
        D.paired_shuffle(srcs, args.cseed)
    c4 = D.c4_lines(3000)
    c4_cal = c4[:2500]
    if args.cseed:
        c4_cal = D.c4_lines(2500, skip=3000 + 2500 * (args.cseed - 1))

    model, tok, meta = load(args.model)
    bos = meta["bos"]
    log("model:", meta)
    dense = MO.dense_copy(model)
    cfg = text_config(model)
    L, N, d = cfg.num_hidden_layers, cfg.intermediate_size, cfg.hidden_size
    total_params = sum(p.numel() for p in model.parameters())

    half = {}
    need = halfsets_for(conds) | {"C4a"}
    H = args.half
    c4_ids = D.pack(tok, c4_cal, 2 * H, bos=bos)
    half["C4a"], half["C4b"] = c4_ids[:H], c4_ids[H:]
    for k, lines in srcs.items():
        if k + "a" in need or k + "b" in need:
            ids = D.pack(tok, lines, 2 * H, bos=bos)
            half[k + "a"], half[k + "b"] = ids[:H], ids[H:]
    if args.theory:
        test_qa = [f"Q: {r['question']} A: {D.ans(r)[0]}" for r in F["test"]]
        test_d = [D.declarative(r["prop"], r["subj"], D.ans(r)[0]) for r in F["test"]]
        body = 512 - (bos is not None)

        def pack_upto(lines, max_seq=128):
            avail = sum(len(tok(t + "\n", add_special_tokens=False)["input_ids"]) for t in lines)
            n = min(max_seq, (avail // body) // 2 * 2)
            return D.pack(tok, lines, n, bos=bos)
        pq, pc = pack_upto(test_qa), pack_upto(test_d)
        hq, hc = pq.shape[0] // 2, pc.shape[0] // 2
        log(f"theory sets: question {pq.shape[0]} x 512 tokens, cloze {pc.shape[0]} x 512 tokens")
        half.update({"Pq_fit": pq[:hq], "Pq_eval": pq[hq:], "Pc_fit": pc[:hc], "Pc_eval": pc[hc:],
                     "C4h": D.pack(tok, c4[2600:3000], 64, bos=bos)})
        need |= {"Pq_fit", "Pq_eval", "Pc_fit", "Pc_eval", "C4h"}
    need = sorted(h for h in need if h in half)
    log("half-sets:", need)

    # ---------------- GRIDE widths (cached per tag)
    gpath = out_dir / f"{args.tag}_gride.json"
    if args.dims_file:
        dims = json.loads(open(args.dims_file).read())["dims"]
        if not gpath.exists():
            gpath.write_text(json.dumps({"model": args.model, "dims": dims, "from": args.dims_file}))
    elif gpath.exists():
        dims = json.loads(gpath.read_text())["dims"]
    elif args.gride_from:
        dims = json.loads((out_dir / f"{args.gride_from}_gride.json").read_text())["dims"]
        gpath.write_text(json.dumps({"model": args.model, "dims": dims, "from": args.gride_from}))
    else:
        dims = MO.gride_dims(MO.collect_clouds(model, half["C4a"]))
        gpath.write_text(json.dumps({"model": args.model, "dims": dims}))
    assert len(dims) == L
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
    todo = sorted(theory_layers) if args.theory_only else list(range(L))
    for g0 in range(0, len(todo), args.group):
        layers = todo[g0:g0 + args.group]
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
                if args.theory_only:                   # nothing is evaluated, so the consumers are not kept
                    for key in [k for k in store if k[0] == b["target"]]:
                        store[key].pop(i, None)
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
    pp = E.ppl_sets(tok, c4[2600:3000], bos=bos)
    zs_conds = {c for c in args.zeroshot.split(",") if c}
    zs = E.zeroshot_sets() if zs_conds else None
    bits = json.loads(bits_path.read_text()) if bits_path.exists() else {}

    def evaluate(name: str, b: dict | None, spec: tuple | None):
        t = time.time()
        rec = {"tag": args.tag, "model": meta["repo"], "rev": meta["rev"], "cseed": args.cseed, "cond": name,
               "target": b["target"] if b else 1.0, "alpha": b["alpha"] if b else None,
               "ffn_keep": b["ffn_keep"] if b else 1.0, "param_keep": b["param_keep"] if b else 1.0,
               "batch": args.batch, "bos": bos}
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
            rec["arc_easy"] = E.multiple_choice(model, tok, zs["arc_easy"], length_norm=False, bos=bos)
            rec["hellaswag"] = E.multiple_choice(model, tok, zs["hellaswag"], length_norm=True, bos=bos)
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
