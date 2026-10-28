"""Sequential-compression pilot.

Usage:  python scripts/experiments/sequential.py <tag>      (qwen15b, olmo1b; qwen7b, olmo7b)
- For each budget and condition the model is compressed block by block: the selection and repair moments of block l
  are collected on the outputs of blocks 0 to l-1 already compressed under the same condition, and block l is then
  compressed before block l+1 is measured. The ridge repair fits the dense down_proj on these inputs. Everything else
  (widths, activation-norm selection, ridge 1e-4, half-sets, prompts, batch, scoring) is as in the paper.
- Conditions: C4|C4, C4|MIX:KG-D, MIX:KG-D|MIX:KG-D, MIX:KG-D~|MIX:KG-D~ (paired file of data/calib_paired), 70% and 50%.
- Compared with the paper's dense-statistics runs of the same conditions in the analysis.
Outputs: results/sequential/<tag>_seq.jsonl and <tag>_seq_bits.json.
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
KEEP = ("C4|C4", "C4|MIX:KG-D", "MIX:KG-D|MIX:KG-D", "MIX:KG-D~|MIX:KG-D~")
H = 64


def main():
    tag = sys.argv[1]
    t0 = time.time()
    log = lambda *a: print(time.strftime("%H:%M:%S"), f"[{tag}_seq]", *a, flush=True)
    out_dir = P.SEQUENTIAL
    out_dir.mkdir(parents=True, exist_ok=True)
    res_path, bits_path = out_dir / f"{tag}_seq.jsonl", out_dir / f"{tag}_seq_bits.json"
    rows = D.load_popqa()
    F = D.facts_by_split(rows, D.make_split(rows))
    conds = [c for c in R.grid("core") if c[0] in KEEP]
    assert sorted(c[0] for c in conds) == sorted(KEEP)
    srcs = {"KG-D": [json.loads(l) for l in open(D.CALIB / "KG-D.jsonl")],
            "KG-D~": [json.loads(l) for l in open(P.calib_file("KG-D~"))]}
    c4 = D.c4_lines(3000)
    model, tok = MO.load(MODELS[tag])
    dense = MO.dense_copy(model)
    L, N = model.config.num_hidden_layers, model.config.intermediate_size
    c4_ids = D.pack(tok, c4[:2500], 2 * H)
    half = {"C4a": c4_ids[:H], "C4b": c4_ids[H:]}
    for k, lines in srcs.items():
        ids = D.pack(tok, lines, 2 * H)
        half[k + "a"], half[k + "b"] = ids[:H], ids[H:]
    dims = json.loads((P.GRID / f"{tag}_gride.json").read_text())["dims"]
    ex = E.shots(F["cal"])
    pp = E.ppl_sets(tok, c4[2600:3000])
    bits = json.loads(bits_path.read_text()) if bits_path.exists() else {}
    for f in (0.7, 0.5):
        widths = MO.widths(dims, MO.alpha_for_fraction(dims, N, f), N)
        for spec in conds:
            cname, rule, ssrc, rep, rsrc = spec
            key = f"{f}|{cname}"
            if key in bits:
                continue
            t = time.time()
            need = sorted(h for h in R.halfsets_for([spec]) if h in half)
            for i in range(L):
                mom = {h: MO.collect_moments(model, half[h], [i]) for h in need}   # inputs from compressed blocks < i
                Wd = dense[i]["down_proj"].float()
                S = C.select(rule, Wd, widths[i], R.moments_of(ssrc, mom, i), dense[i], seed=0)
                A = C.ridge(Wd, S, R.moments_of(rsrc, mom, i), lam_rel=1e-4).to(torch.bfloat16)
                MO.apply(model, dense, i, S, A, None)
                del mom
            build = time.time() - t
            vq = E.recall(model, tok, F["val"], ex, "question", BATCH[tag])
            tq = E.recall(model, tok, F["test"], ex, "question", BATCH[tag])
            tc = E.recall(model, tok, F["test"], ex, "cloze", BATCH[tag])
            rec = {"tag": tag, "cond": cname, "target": f, "mode": "sequential", "val_q_acc": sum(vq) / len(vq),
                   "test_q_acc": sum(tq) / len(tq), "test_c_acc": sum(tc) / len(tc),
                   "ppl_wikitext2": E.perplexity(model, pp["wikitext2"]), "ppl_c4": E.perplexity(model, pp["c4"]),
                   "build_secs": round(build, 1), "secs": round(time.time() - t, 1)}
            bits[key] = {k: "".join("1" if x else "0" for x in v) for k, v in (("val_q", vq), ("test_q", tq), ("test_c", tc))}
            with open(res_path, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            bits_path.write_text(json.dumps(bits))
            log(f"{key:28s} test_q={rec['test_q_acc']:.4f} test_c={rec['test_c_acc']:.4f} "
                f"ppl_wt={rec['ppl_wikitext2']:.2f} (build {build:.0f}s, total {rec['secs']}s)")
            MO.restore(model, dense)
    log(f"all done in {time.time() - t0:.0f}s")
    print("JOB OK", tag, flush=True)


if __name__ == "__main__":
    main()
