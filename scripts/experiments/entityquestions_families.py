"""EntityQuestions on the further families: entityquestions.py with the kgx package.

Differences from entityquestions.py, and nothing else: models from kgx.models.REGISTRY at pinned revisions; every
calibration sequence starts with BOS for BOS families (kgx.data.pack); the replay reference is the family's main run
(results/families/<tag>_bits.json); outputs go to results/entityquestions/<tag>_eq*.

Usage:  python scripts/experiments/entityquestions_families.py <tag> [b [cseed]]   (cseed 1, 2: truth conditions on other samples)
Mode b (replication): calibration on the EntityQuestions graph (data/calib_eq: EQ-D, EQ-D~, EQ-QA, EQ-QA~,
paired line by line); conditions C4|C4 (must reproduce mode a's EntityQuestions bits), C4|MIX:EQ-D, MIX:EQ-D|none and
both channels on MIX(EQ-D), MIX(EQ-D~), MIX(EQ-QA), MIX(EQ-QA~); PopQA is not scored. Outputs <tag>_eqb*.
- Conditions, calibration and widths exactly as in the paper (sample 0): none|C4, C4|C4, C4|MIX:KG-D, MIX:KG-D|none,
  MIX:KG-D|MIX:KG-D, MIX:KG-D~|MIX:KG-D~ (the paired counterfactual file of data/calib_paired). Moments, selection and
  ridge come from kgcal.run_grid (moments_of, halfsets_for, grid), so each compressed model is that of the main run.
- Scored on data/entityquestions/eq_test_sample.json with the paper's question prompt (four PopQA calibration
  examples) and alias rule, 12 greedy tokens. PopQA test questions are also scored for every condition and must
  reproduce the paper's bits exactly (replay check, asserted at the end).
- Outputs: results/entityquestions/<tag>_eq.jsonl, <tag>_eq_bits.json (eq and popqa bits), <tag>_eq_answers.json.
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
from kgx import data as D              # noqa: E402
from kgx import evaluate as E          # noqa: E402
from kgx import model_ops as MO        # noqa: E402
from kgx.models import load, text_config  # noqa: E402
from kgcal import run_grid as R        # noqa: E402

MODELS = {"llama8b": "llama8b", "mistral7b": "mistral7b", "phi4": "phi4"}   # kgx.models.REGISTRY tags
BATCH = {"llama8b": 64, "mistral7b": 64, "phi4": 64}
GROUP = {"llama8b": 2, "mistral7b": 2, "phi4": 1}
KEEP = ("none|C4", "C4|C4", "C4|MIX:KG-D", "MIX:KG-D|none", "MIX:KG-D|MIX:KG-D", "MIX:KG-D~|MIX:KG-D~")
H = 64


@torch.no_grad()
def answers(model, tok, facts, examples, batch, max_new=12):
    """kgcal.evaluate.recall with the decoded continuation kept."""
    dev = next(model.parameters()).device
    tok.padding_side = "left"
    ps = E.prompts(facts, examples, "question")
    out, texts = [], []
    for b in range(0, len(ps), batch):
        enc = tok(ps[b:b + batch], return_tensors="pt", padding=True).to(dev)
        g = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        for r, t in zip(facts[b:b + batch], tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
            line = t.strip().split("\n")[0].lower()
            out.append(any(a.lower() in line for a in D.ans(r)))
            texts.append(t)
    return out, texts


def main():
    tag = sys.argv[1]
    mode_b = len(sys.argv) > 2 and sys.argv[2] == "b"
    cseed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    stem = (f"{tag}_eqb" + (f"_cs{cseed}" if cseed else "")) if mode_b else f"{tag}_eq"
    t0 = time.time()
    log = lambda *a: print(time.strftime("%H:%M:%S"), f"[{stem}]", *a, flush=True)
    out_dir = P.EQ_RESULTS
    out_dir.mkdir(parents=True, exist_ok=True)
    res_path, bits_path, ans_path = (out_dir / f"{stem}{s}" for s in (".jsonl", "_bits.json", "_answers.json"))
    rows = D.load_popqa()
    F = D.facts_by_split(rows, D.make_split(rows))
    EQ = json.loads((P.EQ / "eq_test_sample.json").read_text())["facts"]
    if mode_b:
        conds = [("C4|C4", "actnorm", "C4", "ridge", "C4"), ("C4|MIX:EQ-D", "actnorm", "C4", "ridge", "MIX:EQ-D"),
                 ("MIX:EQ-D|none", "actnorm", "MIX:EQ-D", "none", None)]
        conds += [(f"MIX:{x}|MIX:{x}", "actnorm", f"MIX:{x}", "ridge", f"MIX:{x}") for x in ("EQ-D", "EQ-D~", "EQ-QA", "EQ-QA~")]
        srcs = {x: [json.loads(l) for l in open(P.CALIB_EQ / f"{x}.jsonl")] for x in ("EQ-D", "EQ-D~", "EQ-QA", "EQ-QA~")}
        if cseed:
            # other calibration samples: truth conditions only; every file is shuffled with the same
            # permutation, so true and counterfactual lines stay paired; C4 is a disjoint draw, as in run_grid
            conds = [c for c in conds if c[0].startswith("MIX:EQ-") and c[0].count("MIX:") == 2]
            for x in srcs:
                random.Random(cseed).shuffle(srcs[x])
    else:
        conds = [c for c in R.grid("core") if c[0] in KEEP]
        assert sorted(c[0] for c in conds) == sorted(KEEP)
        srcs = {"KG-D": [json.loads(l) for l in open(D.calib_path("KG-D"))],
                "KG-D~": [json.loads(l) for l in open(D.calib_path("KG-D~"))]}
    c4 = D.c4_lines(3000)
    model, tok, meta = load(MODELS[tag])
    bos = meta["bos"]
    dense = MO.dense_copy(model)
    cfg = text_config(model)
    L, N, d = cfg.num_hidden_layers, cfg.intermediate_size, cfg.hidden_size
    total_params = sum(p.numel() for p in model.parameters())
    c4_ids = D.pack(tok, D.c4_lines(2500, skip=3000 + 2500 * (cseed - 1)) if cseed else c4[:2500], 2 * H, bos=bos)
    half = {"C4a": c4_ids[:H], "C4b": c4_ids[H:]}
    for k, lines in srcs.items():
        ids = D.pack(tok, lines, 2 * H, bos=bos)
        half[k + "a"], half[k + "b"] = ids[:H], ids[H:]
    need = sorted(h for h in R.halfsets_for(conds) if h in half)
    dims = json.loads((P.FAMILIES / f"{tag}_gride.json").read_text())["dims"]
    budgets = []
    for f in (0.7, 0.5):
        a = MO.alpha_for_fraction(dims, N, f)
        w = MO.widths(dims, a, N)
        removed = sum((N - m) * 3 * d for m in w)
        budgets.append({"target": f, "widths": w, "ffn_keep": sum(w) / (L * N),
                        "param_keep": (total_params - removed) / total_params})
    store = {}
    for g0 in range(0, L, GROUP[tag]):
        layers = list(range(g0, min(L, g0 + GROUP[tag])))
        mom = {h: MO.collect_moments(model, half[h], layers) for h in need}
        for b in budgets:
            for i in layers:
                Wd = dense[i]["down_proj"].float()
                sel = {}
                for cname, rule, ssrc, rep, rsrc in conds:
                    if (rule, ssrc) not in sel:
                        sel[(rule, ssrc)] = C.select(rule, Wd, b["widths"][i], R.moments_of(ssrc, mom, i), dense[i], seed=0)
                    S = sel[(rule, ssrc)]
                    A = None if rep == "none" else C.ridge(Wd, S, R.moments_of(rsrc, mom, i), lam_rel=1e-4).to(torch.bfloat16)
                    store.setdefault((b["target"], cname), {})[i] = (S, A, None)
        del mom
        torch.cuda.empty_cache()
        log(f"layers {layers[0]}-{layers[-1]} done ({time.time() - t0:.0f}s)")

    ex = E.shots(F["cal"])
    bits = json.loads(bits_path.read_text()) if bits_path.exists() else {}
    ANS = json.loads(ans_path.read_text()) if ans_path.exists() else {}

    def evaluate(key, b):
        t = time.time()
        eq, eq_txt = answers(model, tok, EQ, ex, BATCH[tag])
        pq = [] if mode_b else E.recall(model, tok, F["test"], ex, "question", BATCH[tag])
        bits[key] = {"eq": "".join("1" if x else "0" for x in eq), "test_q": "".join("1" if x else "0" for x in pq)}
        ANS[key] = eq_txt
        rec = {"tag": tag, "key": key, "eq_acc": sum(eq) / len(eq), "test_q_acc": sum(pq) / len(pq) if pq else None,
               "param_keep": b["param_keep"] if b else 1.0, "secs": round(time.time() - t, 1)}
        with open(res_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        bits_path.write_text(json.dumps(bits))
        ans_path.write_text(json.dumps(ANS))
        log(f"{key:28s} eq={rec['eq_acc']:.4f} popqa={rec['test_q_acc']} ({rec['secs']}s)")

    if "1.0|dense" not in bits and not mode_b:
        evaluate("1.0|dense", None)
    for b in budgets:
        for cname, *_ in conds:
            key = f"{b['target']}|{cname}"
            if key in bits:
                continue
            for i in range(L):
                MO.apply(model, dense, i, *store[(b["target"], cname)][i])
            evaluate(key, b)
            MO.restore(model, dense)
    if mode_b and cseed:
        log(f"all done in {time.time() - t0:.0f}s")
        print("JOB OK", stem, flush=True)
        return
    if mode_b:
        a = json.loads((out_dir / f"{tag}_eq_bits.json").read_text())
        for key in ("0.7|C4|C4", "0.5|C4|C4"):
            log(f"replay {key} vs mode a: {'identical' if bits[key]['eq'] == a[key]['eq'] else 'DIFFERENT'}")
            assert bits[key]["eq"] == a[key]["eq"], key
        log(f"all done in {time.time() - t0:.0f}s")
        print("JOB OK", stem, flush=True)
        return
    ref = json.loads((P.FAMILIES / f"{tag}_bits.json").read_text())
    pc = ref                                      # the extension's main run holds the paired variants too
    for key in bits:
        theirs = (pc if "~" in key else ref)[key]["test_q"]
        log(f"replay {key}: {'identical' if bits[key]['test_q'] == theirs else 'DIFFERENT'}")
        assert bits[key]["test_q"] == theirs, key
    log(f"all done in {time.time() - t0:.0f}s")
    print("JOB OK", tag, flush=True)


if __name__ == "__main__":
    main()
