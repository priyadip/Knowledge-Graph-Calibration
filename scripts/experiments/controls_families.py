"""Controls for format, scoring and truth on the further families: controls.py with the kgx package.

Usage:  python scripts/experiments/controls_families.py <llama8b|mistral7b|phi4> <fmt|mc|truth> [cseed]
Differences from controls.py, and nothing else:
- models from kgx.models (pinned revisions; Phi-4's fused gate and up projections split into two) and widths from the
  family's GRIDE files (results/families/<tag>_gride.json);
- for a family with a start-of-text token (Llama-3.1-8B, Mistral-7B-v0.3) every calibration sequence is that token
  and 511 text tokens, as in kgx.data.pack: the C4 halves and the halves of the paired sources are packed by
  kgx.data.pack, and the exact-fact halves of fmt and truth are cut into chunks of 511 text tokens, each preceded by
  the token, so a half of 64 sequences holds 64 x 511 text tokens; perplexity windows as in kgx.evaluate.ppl_sets;
- multiple choice: candidate tokens are the tokens that follow the prompt when prompt, a space and the candidate are
  tokenized together (the prompt's own tokens must be a prefix; otherwise the candidate is tokenized alone and the
  case is counted), since a SentencePiece tokenizer adds a spurious word-start token to a candidate tokenized alone;
- replay references are the family's main runs (results/families/<tag>_bits.json), except Llama-3.1-8B with C4
  selection at 70%, where the EntityQuestions reruns, which identical reruns reproduce, are the reference.
Outputs: results/controls/<tag>_<exp>.jsonl, _bits.json, _info.json (the info file is written after all replays pass).
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
from kgcal import run_grid as R        # noqa: E402
from kgx import data as D              # noqa: E402
from kgx import evaluate as E          # noqa: E402
from kgx import model_ops as MO        # noqa: E402
from kgx.models import load, text_config  # noqa: E402

MODELS = {"llama8b": "llama8b", "mistral7b": "mistral7b", "phi4": "phi4"}
BATCH = {"llama8b": 64, "mistral7b": 64, "phi4": 64}
MCBATCH = {"llama8b": 16, "mistral7b": 16, "phi4": 8}
GROUP = {"llama8b": 2, "mistral7b": 2, "phi4": 1}
H, SEQ = 64, 512
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


def moments_var(model, ids, layers, bos):
    """Token-weighted moments over full chunks and the last partial chunk (no padding); with a start-of-text token,
    every chunk is that token and up to 511 text tokens, as in kgx.data.pack."""
    body = SEQ if bos is None else SEQ - 1
    n_full = len(ids) // body
    parts = []
    if n_full:
        parts.append(torch.tensor(ids[:n_full * body]).view(n_full, body))
    if len(ids) > n_full * body:
        rem = ids[n_full * body:]
        parts.append(torch.tensor(rem).view(1, len(rem)))
    if bos is not None:
        parts = [torch.cat([torch.full((t.shape[0], 1), bos, dtype=t.dtype), t], 1) for t in parts]
    acc, tot = None, 0
    for t in parts:
        n = t.numel()
        m = MO.collect_moments(model, t, layers)
        acc = {i: m[i] * n for i in layers} if acc is None else {i: acc[i] + m[i] * n for i in layers}
        tot += n
    return {i: acc[i] / tot for i in layers}


@torch.no_grad()
def mc_eval(model, seqs, n_items, n_cands, batch, pad):
    """seqs: list of (item, cand, ids, prompt_len); returns bool per item: gold (cand 0) scores strictly highest."""
    dev = model.get_input_embeddings().weight.device
    order = sorted(range(len(seqs)), key=lambda k: len(seqs[k][2]))
    score = np.full((n_items, n_cands), -np.inf)
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
            tgt = torch.tensor(ids[lp:], device=logits.device)
            lg = logits[r, lp - 1:len(ids) - 1].float().log_softmax(-1)
            score[item, cand] = float(lg.gather(1, tgt[:, None]).mean())
        del logits
    return score[:, 0] > score[:, 1:].max(1)


def main():
    tag, exp = sys.argv[1], sys.argv[2]
    cseed = int(sys.argv[3]) if len(sys.argv) > 3 else 0     # calibration samples 1 and 2, truth only
    assert cseed == 0 or exp == "truth", "other calibration samples are run for the truth experiment only"
    t0 = time.time()
    log = lambda *a: print(time.strftime("%H:%M:%S"), f"[{tag}_{exp}{'_cs' + str(cseed) if cseed else ''}]", *a, flush=True)
    out_dir = P.CONTROL_RESULTS
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / (f"{tag}_{exp}" + (f"_cs{cseed}" if cseed else ""))
    rows = D.load_popqa()
    F = D.facts_by_split(rows, D.make_split(rows))
    ex = E.shots(F["cal"])
    c4 = D.c4_lines(3000)
    model, tok, meta = load(MODELS[tag])
    bos = meta["bos"]
    dense = MO.dense_copy(model)
    cfg = text_config(model)
    L, N = cfg.num_hidden_layers, cfg.intermediate_size
    kg_tokens = H * (SEQ if bos is None else SEQ - 1)
    c4_cal = c4[:2500] if not cseed else D.c4_lines(2500, skip=3000 + 2500 * (cseed - 1))   # as kgcal.run_grid
    c4_ids = D.pack(tok, c4_cal, 2 * H, bos=bos)
    halves = {"C4a": c4_ids[:H], "C4b": c4_ids[H:]}
    var = {}
    info = {"tag": tag, "exp": exp, "bos": bos, "kg_text_tokens": kg_tokens}
    read = lambda p: [json.loads(l) for l in open(p)]
    if exp == "fmt":
        fmt = json.loads((P.CONTROLS / "fmt_facts.json").read_text())
        qa_ids = line_ids(tok, [f["qa"] for f in fmt])
        k = whole_prefix(qa_ids, kg_tokens)
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
        true = read(D.calib_path("KG-D"))
        cf = read(D.calib_path("KG-D~"))
        tv = json.loads((P.CONTROLS / "truth_variants.json").read_text())
        if cseed:                  # the fact order of calibration sample cseed (kgcal.run_grid shuffles KG-D first; paired
            perm = list(range(len(true)))                    # variants take the same permutation, kgx.data.paired_shuffle)
            random.Random(cseed).shuffle(perm)
            true, cf = [true[i] for i in perm], [cf[i] for i in perm]
            tv = {kk: [vv[i] for i in perm] for kk, vv in tv.items()}
        k = whole_prefix(line_ids(tok, true), kg_tokens)
        src = {"V0": true[:k], "V100": cf[:k], "V25": tv["V25"][:k], "V50": tv["V50"][:k], "V75": tv["V75"][:k],
               "VX": tv["VX"][:k], "VC": [l for pair in tv["VC_pairs"][:k] for l in pair]}
        for n, lines in src.items():
            var[n] = flat(line_ids(tok, lines))
        info.update({"facts": k, "tokens": {n: len(v) for n, v in var.items()}})
        conds = [("C4|C4", "C4", "C4")] + [(n, n, n) for n in ("V0", "V25", "V50", "V75", "V100", "VX", "VC")]
    else:
        for n in ("KG-D", "KG-QA", "KG-D~", "KG-QA~"):           # MIX(X) uses only the first half of X
            halves[n + "a"] = D.pack(tok, read(D.calib_path(n)), 2 * H, bos=bos)[:H]
        conds = [c for c in R.grid("core") if c[0] in MC_CONDS]
        assert sorted(c[0] for c in conds) == sorted(MC_CONDS)
        mc = json.loads((P.CONTROLS / "test_mc.json").read_text())
        prompts = E.prompts(F["test"], ex, "question")
        seqs, alone = [], 0
        for it in mc:
            p = tok(prompts[it["i"]])["input_ids"]
            for ci, cand in enumerate(it["cands"]):
                joint = tok(prompts[it["i"]] + " " + cand)["input_ids"]
                if joint[:len(p)] == p and len(joint) > len(p):
                    seqs.append((it["i"], ci, joint, len(p)))
                else:
                    alone += 1
                    seqs.append((it["i"], ci, p + tok(" " + cand, add_special_tokens=False)["input_ids"], len(p)))
        info.update({"mc_items": len(mc), "mc_seqs": len(seqs), "mc_candidates_tokenized_alone": alone})
    log("info:", json.dumps(info))

    dims = json.loads((P.FAMILIES / f"{tag}_gride.json").read_text())["dims"]
    budgets = []
    for f in (0.7, 0.5):
        a = MO.alpha_for_fraction(dims, N, f)
        budgets.append({"target": f, "widths": MO.widths(dims, a, N)})
    store = {}
    for g0 in range(0, L, GROUP[tag]):
        layers = list(range(g0, min(L, g0 + GROUP[tag])))
        mom = {h: MO.collect_moments(model, t, layers) for h, t in halves.items()}
        for h, ids in var.items():
            mom[h] = moments_var(model, ids, layers, bos)

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

    pp = E.ppl_sets(tok, c4[2600:3000], bos=bos)
    bits, recs = {}, []
    mc_seqs = seqs if exp == "mc" else None

    def evaluate(key):
        t = time.time()
        rec = {"tag": tag, "exp": exp, "key": key}
        if exp == "mc":
            ok = mc_eval(model, mc_seqs, len(F["test"]), 4, MCBATCH[tag], tok.pad_token_id)
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
    ref = json.loads((P.FAMILIES / f"{tag}_bits.json").read_text())
    ref_cs = json.loads((P.FAMILIES / f"{tag}_cs{cseed}_bits.json").read_text()) if cseed else None
    if tag == "llama8b" and not cseed:                       # C4 selection at 70% reproduces the EntityQuestions reruns, not the original
        rerun = json.loads((P.EQ_RESULTS / "llama8b_eq_bits.json").read_text())
        ref = dict(ref, **{"0.7|C4|C4": rerun["0.7|C4|C4"]})
    for key in [k for k in bits if "test_q" in bits[k] and (k == "1.0|dense" or k.endswith("|C4|C4") or k.endswith("|MIX:KG-D|MIX:KG-D"))]:
        reference = ref_cs if (cseed and key != "1.0|dense") else ref
        same = bits[key]["test_q"] == reference[key]["test_q"]
        log(f"replay {key}: {'identical' if same else 'DIFFERENT'}")
        if tag == "llama8b" and cseed and key == "0.7|C4|C4":
            continue                  # reruns of this condition differ on a few answers (see the paper, Appendix C)
        assert same, key
    Path(f"{stem}_info.json").write_text(json.dumps(info, indent=1))
    log(f"all done in {time.time() - t0:.0f}s")
    print("JOB OK", tag, exp, flush=True)


if __name__ == "__main__":
    main()
