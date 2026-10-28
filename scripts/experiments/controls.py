"""Controls for format, scoring and truth: matched format, likelihood multiple choice, truth variants.

Usage:  python scripts/experiments/controls.py <tag> <fmt|mc|truth> [cseed]     (Qwen2.5 and OLMo-2 models)
Everything not named here is as in the paper (kgcal): GRIDE widths of the main runs, activation-norm selection, ridge
1e-4, C4 halves C4a and C4b, four calibration examples, 12 greedy tokens, alias matching.
- fmt:   both channels on MIX(QA(S)), MIX(D(S)) and MIX(D(S)rep), and C4/C4 (replay). S: the facts whose
         question-answer lines of the paper's KG-QA source fit in 32,768 tokens, whole lines, under this tokenizer.
- truth: both channels on MIX(V) for V0, V25, V50, V75, V100, VX, VC (data/controls/truth_variants.json), and C4/C4.
         S: the facts whose true lines of the paper's KG-D source fit in 32,768 tokens, whole lines.
- mc:    the nine core conditions, scored by likelihood multiple choice (data/controls/test_mc.json); C4/C4 and
         MIX(KG-D)/MIX(KG-D) are also scored by generation and must reproduce the paper.
Halves with an exact fact set are tokenized line by line (each line and a newline, as in kgcal.data.pack) and their
moments are token-weighted means over the full 512-token chunks and the last partial chunk, so no padding enters them.
MIX(X) averages the moments of C4a and of X with equal weight, as kgcal.run_grid.moments_of does.
cseed 1, 2 (truth only): other calibration samples. KGCAL_GEN_BATCH overrides the generation batch size (the
Qwen2.5-7B and OLMo-2-7B sample-2 runs used 48, as their main runs did).
Outputs: results/controls/<tag>_<exp>.jsonl, _bits.json, _info.json.
"""
import json
import random
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
from kgcal import evaluate as E        # noqa: E402
from kgcal import model_ops as MO      # noqa: E402
from kgcal import run_grid as R        # noqa: E402

MODELS = {"qwen15b": "Qwen/Qwen2.5-1.5B", "olmo1b": "allenai/OLMo-2-0425-1B",
          "qwen7b": "Qwen/Qwen2.5-7B", "olmo7b": "allenai/OLMo-2-1124-7B"}
BATCH = {"qwen15b": 96, "olmo1b": 96, "qwen7b": 64, "olmo7b": 64}
MCBATCH = {"qwen15b": 32, "olmo1b": 32, "qwen7b": 16, "olmo7b": 16}
GROUP = {"qwen15b": 4, "olmo1b": 4, "qwen7b": 1, "olmo7b": 1}
H, KG_TOKENS, SEQ = 64, 64 * 512, 512
MC_CONDS = ("none|C4", "C4|C4", "C4|MIX:KG-D", "MIX:KG-D|C4", "MIX:KG-D|none", "MIX:KG-D|MIX:KG-D",
            "MIX:KG-D~|MIX:KG-D~", "MIX:KG-QA|MIX:KG-QA", "MIX:KG-QA~|MIX:KG-QA~")


def line_ids(tok, lines):
    return [tok(t + "\n", add_special_tokens=False)["input_ids"] for t in lines]


def whole_prefix(ids_per_line, max_tokens):
    """Number of leading lines whose tokens fit in max_tokens."""
    tot = 0
    for k, ids in enumerate(ids_per_line):
        if tot + len(ids) > max_tokens:
            return k
        tot += len(ids)
    return len(ids_per_line)


def flat(ids_per_line):
    return [t for ids in ids_per_line for t in ids]


def moments_var(model, ids, layers):
    """Token-weighted moments over full chunks and the last partial chunk (no padding)."""
    n_full = len(ids) // SEQ
    parts = []
    if n_full:
        parts.append((torch.tensor(ids[:n_full * SEQ]).view(n_full, SEQ), n_full * SEQ))
    if len(ids) > n_full * SEQ:
        rem = ids[n_full * SEQ:]
        parts.append((torch.tensor(rem).view(1, len(rem)), len(rem)))
    acc, tot = None, 0
    for t, n in parts:
        m = MO.collect_moments(model, t, layers)
        acc = {i: m[i] * n for i in layers} if acc is None else {i: acc[i] + m[i] * n for i in layers}
        tot += n
    return {i: acc[i] / tot for i in layers}


@torch.no_grad()
def mc_eval(model, seqs, n_items, n_cands, batch):
    """seqs: list of (item, cand, ids, prompt_len); returns bool per item: gold (cand 0) scores strictly highest."""
    dev = next(model.parameters()).device
    order = sorted(range(len(seqs)), key=lambda k: len(seqs[k][2]))
    score = np.full((n_items, n_cands), -np.inf)
    pad = 0
    for b0 in range(0, len(order), batch):
        idx = order[b0:b0 + batch]
        L = max(len(seqs[k][2]) for k in idx)
        x = torch.full((len(idx), L), pad, dtype=torch.long)
        att = torch.zeros((len(idx), L), dtype=torch.long)
        for r, k in enumerate(idx):
            ids = seqs[k][2]
            x[r, :len(ids)] = torch.tensor(ids)
            att[r, :len(ids)] = 1
        logits = model(input_ids=x.to(dev), attention_mask=att.to(dev)).logits
        for r, k in enumerate(idx):
            item, cand, ids, lp = seqs[k]
            tgt = torch.tensor(ids[lp:], device=dev)
            lg = logits[r, lp - 1:len(ids) - 1].float().log_softmax(-1)
            score[item, cand] = float(lg.gather(1, tgt[:, None]).mean())
        del logits
    return score[:, 0] > score[:, 1:].max(1)


def main():
    tag, exp = sys.argv[1], sys.argv[2]
    cseed = int(sys.argv[3]) if len(sys.argv) > 3 else 0     # calibration samples 1 and 2, truth only
    assert cseed == 0 or exp == "truth", "other calibration samples are run for the truth experiment only"
    if "KGCAL_GEN_BATCH" in __import__("os").environ:     # the stored 7B sample-2 runs used 48
        BATCH[tag] = int(__import__("os").environ["KGCAL_GEN_BATCH"])
    t0 = time.time()
    log = lambda *a: print(time.strftime("%H:%M:%S"), f"[{tag}_{exp}{'_cs' + str(cseed) if cseed else ''}]", *a, flush=True)
    out_dir = P.CONTROL_RESULTS
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / (f"{tag}_{exp}" + (f"_cs{cseed}" if cseed else ""))
    rows = D.load_popqa()
    F = D.facts_by_split(rows, D.make_split(rows))
    ex = E.shots(F["cal"])
    c4 = D.c4_lines(3000)
    model, tok = MO.load(MODELS[tag])
    dense = MO.dense_copy(model)
    L, N, d = model.config.num_hidden_layers, model.config.intermediate_size, model.config.hidden_size
    c4_cal = c4[:2500] if not cseed else D.c4_lines(2500, skip=3000 + 2500 * (cseed - 1))   # as kgcal.run_grid
    c4_ids = D.pack(tok, c4_cal, 2 * H)
    halves = {"C4a": c4_ids[:H], "C4b": c4_ids[H:]}         # tensors (paper packing)
    var = {}                                                 # name -> flat token list (exact fact sets)
    info = {"tag": tag, "exp": exp}
    read = lambda p: [json.loads(l) for l in open(p)]
    if exp == "fmt":
        fmt = json.loads((P.CONTROLS / "fmt_facts.json").read_text())
        qa_ids = line_ids(tok, [f["qa"] for f in fmt])
        k = whole_prefix(qa_ids, KG_TOKENS)
        dl_ids = line_ids(tok, [f["decl"] for f in fmt[:k]])
        var["QA_S"], var["D_S"] = flat(qa_ids[:k]), flat(dl_ids)
        rep, j = [], 0
        while len(rep) + len(dl_ids[j % k]) <= len(var["QA_S"]):
            rep.extend(dl_ids[j % k]); j += 1
        var["D_Srep"] = rep
        info.update({"facts": k, "tokens": {n: len(v) for n, v in var.items()}, "D_Srep_lines": j,
                     "D_Srep_facts_twice": j - k})
        conds = [("C4|C4", "C4", "C4")] + [(n, n, n) for n in ("QA_S", "D_S", "D_Srep")]
    elif exp == "truth":
        true = read(D.CALIB / "KG-D.jsonl")
        cf = read(P.calib_file("KG-D~"))
        tv = json.loads((P.CONTROLS / "truth_variants.json").read_text())
        if cseed:                  # the fact order of calibration sample cseed (kgcal.run_grid shuffles KG-D first; paired
            perm = list(range(len(true)))                    # variants take the same permutation)
            random.Random(cseed).shuffle(perm)
            true, cf = [true[i] for i in perm], [cf[i] for i in perm]
            tv = {kk: [vv[i] for i in perm] for kk, vv in tv.items()}
        k = whole_prefix(line_ids(tok, true), KG_TOKENS)
        src = {"V0": true[:k], "V100": cf[:k], "V25": tv["V25"][:k], "V50": tv["V50"][:k], "V75": tv["V75"][:k],
               "VX": tv["VX"][:k], "VC": [l for pair in tv["VC_pairs"][:k] for l in pair]}
        for n, lines in src.items():
            var[n] = flat(line_ids(tok, lines))
        info.update({"facts": k, "tokens": {n: len(v) for n, v in var.items()}})
        conds = [("C4|C4", "C4", "C4")] + [(n, n, n) for n in ("V0", "V25", "V50", "V75", "V100", "VX", "VC")]
    else:
        srcs = {"KG-D": read(D.CALIB / "KG-D.jsonl"), "KG-QA": read(D.CALIB / "KG-QA.jsonl"),
                "KG-D~": read(P.calib_file("KG-D~")),
                "KG-QA~": read(P.calib_file("KG-QA~"))}
        for n, lines in srcs.items():                        # MIX(X) uses only the first half of X
            halves[n + "a"] = D.pack(tok, lines, 2 * H)[:H]
        conds = [c for c in R.grid("core") if c[0] in MC_CONDS]
        assert sorted(c[0] for c in conds) == sorted(MC_CONDS)
        mc = json.loads((P.CONTROLS / "test_mc.json").read_text())
        prompts = E.prompts(F["test"], ex, "question")
        seqs = []
        for it in mc:
            p = tok(prompts[it["i"]])["input_ids"]
            for ci, cand in enumerate(it["cands"]):
                seqs.append((it["i"], ci, p + tok(" " + cand, add_special_tokens=False)["input_ids"], len(p)))
        info.update({"mc_items": len(mc), "mc_seqs": len(seqs)})
    log("info:", json.dumps(info))

    dims = json.loads((P.GRID / f"{tag}_gride.json").read_text())["dims"]
    budgets = []
    for f in (0.7, 0.5):
        a = MO.alpha_for_fraction(dims, N, f)
        budgets.append({"target": f, "widths": MO.widths(dims, a, N)})
    store = {}
    for g0 in range(0, L, GROUP[tag]):
        layers = list(range(g0, min(L, g0 + GROUP[tag])))
        mom = {h: MO.collect_moments(model, t, layers) for h, t in halves.items()}
        for h, ids in var.items():
            mom[h] = moments_var(model, ids, layers)

        def M(src, i):
            if exp == "mc":
                return R.moments_of(src, mom, i)
            if src == "C4":
                return (mom["C4a"][i] + mom["C4b"][i]) / 2
            return (mom["C4a"][i] + mom[src][i]) / 2

        for b in budgets:
            for i in layers:
                Wd = dense[i]["down_proj"].float()
                sel = {}
                for cond in conds:
                    cname, ssrc, rsrc = (cond[0], cond[2], cond[4] if cond[3] == "ridge" else None) if exp == "mc" else cond
                    if ssrc not in sel:
                        sel[ssrc] = C.select("actnorm", Wd, b["widths"][i], M(ssrc, i), dense[i], seed=0)
                    A = None if rsrc is None else C.ridge(Wd, sel[ssrc], M(rsrc, i), lam_rel=1e-4).to(torch.bfloat16)
                    store.setdefault((b["target"], cname), {})[i] = (sel[ssrc], A, None)
        del mom
        torch.cuda.empty_cache()
        log(f"layers {layers[0]}-{layers[-1]} done ({time.time() - t0:.0f}s)")

    pp = E.ppl_sets(tok, c4[2600:3000])
    bits, recs = {}, []
    mc_seqs = seqs if exp == "mc" else None

    def evaluate(key):
        t = time.time()
        rec = {"tag": tag, "exp": exp, "key": key}
        if exp == "mc":
            ok = mc_eval(model, mc_seqs, len(F["test"]), 4, MCBATCH[tag])
            bits[key] = {"mc": "".join("1" if x else "0" for x in ok)}
            rec["mc_acc"] = float(ok.mean())
            if key.endswith(("|C4|C4", "|MIX:KG-D|MIX:KG-D")) or key == "1.0|dense":
                tq = E.recall(model, tok, F["test"], ex, "question", BATCH[tag])
                bits[key]["test_q"] = "".join("1" if x else "0" for x in tq)
        else:
            tq = E.recall(model, tok, F["test"], ex, "question", BATCH[tag])
            tc = E.recall(model, tok, F["test"], ex, "cloze", BATCH[tag])
            bits[key] = {"test_q": "".join("1" if x else "0" for x in tq), "test_c": "".join("1" if x else "0" for x in tc)}
            rec.update({"test_q_acc": sum(tq) / len(tq), "test_c_acc": sum(tc) / len(tc)})
        rec["ppl_wikitext2"] = E.perplexity(model, pp["wikitext2"])
        rec["secs"] = round(time.time() - t, 1)
        recs.append(rec)
        with open(f"{stem}.jsonl", "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        Path(f"{stem}_bits.json").write_text(json.dumps(bits))
        log(f"{key:26s} " + " ".join(f"{k}={v:.4f}" for k, v in rec.items() if k.endswith("acc")) + f" ({rec['secs']}s)")

    evaluate("1.0|dense")
    for b in budgets:
        for cond in conds:
            key = f"{b['target']}|{cond[0]}"
            for i in range(L):
                MO.apply(model, dense, i, *store[(b["target"], cond[0])][i])
            evaluate(key)
            MO.restore(model, dense)
    ref = json.loads((P.GRID / f"{tag}_bits.json").read_text())
    ref_cs = json.loads((P.MERGED / f"{tag}_cs{cseed}_bits.json").read_text()) if cseed else None
    for key in [k for k in bits if "test_q" in bits[k] and (k == "1.0|dense" or k.endswith("|C4|C4") or k.endswith("|MIX:KG-D|MIX:KG-D"))]:
        # a sample run is checked against the stored run of that sample, which holds its own dense answers when it was
        # generated with another batch size (the 7B sample-2 runs, batch 48)
        reference = ref_cs if (cseed and (key != "1.0|dense" or key in ref_cs)) else ref
        same = bits[key]["test_q"] == reference[key]["test_q"]
        log(f"replay {key}: {'identical' if same else 'DIFFERENT'}")
        assert same, key
    Path(f"{stem}_info.json").write_text(json.dumps(info, indent=1))
    log(f"all done in {time.time() - t0:.0f}s")
    print("JOB OK", tag, exp, flush=True)


if __name__ == "__main__":
    main()
