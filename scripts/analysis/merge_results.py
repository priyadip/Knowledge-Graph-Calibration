"""Build results/merged: the result set the paper is generated from.

The grid runs (results/grid, written by kgcal.run_grid) are kept for every condition except the ones that
compare a true calibration text with a controlled variant of the SAME facts:
  - MIX:KG-D~|MIX:KG-D~ and MIX:KG-QA~|MIX:KG-QA~ (counterfactual), seeds 0, 1, 2
  - MIX:KG-S|MIX:KG-S (synthetic subjects), seed 0
For these, the merged file takes the run whose calibration file is paired line by line with its true partner
(results/paired). Preference order per entry:
  1. batch_matched: rerun at the generation batch size of the original true run (exact comparison)
  2. a1_seed0 / a2_seeds / a4_entities: earlier paired runs (a4 and the a2 7B seed-1 and Qwen-1.5B runs already use the
     original batch size; the others differ in batch size only)
Theory rows (Lemma 1 terms) for the counterfactual selections are taken from paired/a3_theory.
The unpaired runs stay in results/grid and are used only for the paired-versus-unpaired comparison.

python scripts/analysis/merge_results.py  ->  results/merged/*, results/merged/MANIFEST.json
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from kgcal import paths as P            # noqa: E402

N = P.ROOT
ORIG, PR, OUT = P.GRID, P.PAIRED, P.MERGED
TAGS = ["qwen15b", "olmo1b", "qwen7b", "olmo7b"]
CF = ["MIX:KG-D~|MIX:KG-D~", "MIX:KG-QA~|MIX:KG-QA~"]
ENT = ["MIX:KG-S|MIX:KG-S"]
BUDGETS = ["0.7", "0.5"]
# generation batch size of the original runs (the true conditions every paired variant is compared with)
ORIG_BATCH = {("qwen15b", 0): 96, ("olmo1b", 0): 96, ("qwen7b", 0): 64, ("olmo7b", 0): 64,
              ("qwen15b", 1): 96, ("olmo1b", 1): 96, ("qwen7b", 1): 64, ("olmo7b", 1): 64,
              ("qwen15b", 2): 96, ("olmo1b", 2): 96, ("qwen7b", 2): 48, ("olmo7b", 2): 48}
# generation batch size used by each earlier paired run
EARLY_BATCH = {("a1", "qwen15b"): 96, ("a1", "olmo1b"): 96, ("a1", "qwen7b"): 32, ("a1", "olmo7b"): 32,
               ("a2", "qwen15b"): 96, ("a2", "olmo1b"): 64, ("a2", "qwen7b"): 64, ("a2", "olmo7b"): 64,
               ("a4", "qwen15b"): 96, ("a4", "olmo1b"): 96, ("a4", "qwen7b"): 64, ("a4", "olmo7b"): 64}


def load(stem):
    j, b = Path(str(stem) + ".jsonl"), Path(str(stem) + "_bits.json")
    if not (j.exists() and b.exists()):
        return None, None
    recs = {}
    for line in open(j):
        r = json.loads(line)
        recs[f"{r['target']}|{r['cond']}"] = r
    return recs, json.loads(b.read_text())


def candidates(tag, seed, kind):
    """Paired sources for (tag, seed, kind) in preference order: (label, stem, batch)."""
    suf = "" if seed == 0 else f"_cs{seed}"
    out = []
    if kind == "cf":
        out.append(("batch_matched", PR / "batch_matched" / f"{tag}_paired{suf}", ORIG_BATCH[(tag, seed)]))
        if seed == 0:
            out.append(("a1_seed0", PR / "a1_seed0" / f"{tag}_aligned", EARLY_BATCH[("a1", tag)]))
        else:
            out.append(("a2_seeds", PR / "a2_seeds" / f"{tag}_paired{suf}", EARLY_BATCH[("a2", tag)]))
    else:
        out.append(("a4_entities", PR / "a4_entities" / f"{tag}_ent", EARLY_BATCH[("a4", tag)]))
    return out


OUT.mkdir(parents=True, exist_ok=True)
manifest = {"entries": [], "replay_checks": []}
for p in ORIG.glob("*"):                                      # start from a copy of every original file
    shutil.copy(p, OUT / p.name)

for tag in TAGS:
    for seed in (0, 1, 2):
        suf = "" if seed == 0 else f"_cs{seed}"
        recs, bits = load(ORIG / f"{tag}{suf}")
        if recs is None:
            continue
        for kind, conds in (("cf", CF), ("ent", ENT if seed == 0 else [])):
            for cond in conds:
                for t in BUDGETS:
                    key = f"{t}|{cond}"
                    for label, stem, batch in candidates(tag, seed, kind):
                        r2, b2 = load(stem)
                        if r2 is None or key not in b2:
                            continue
                        rec = dict(r2[key]); rec["tag"] = tag; rec["paired_source"] = f"{label}:{stem.name}"
                        recs[key], bits[key] = rec, b2[key]
                        matched = batch == ORIG_BATCH[(tag, seed)]
                        manifest["entries"].append({"tag": tag, "seed": seed, "key": key, "source": label,
                                                    "file": str(stem.relative_to(N)), "batch": batch,
                                                    "orig_batch": ORIG_BATCH[(tag, seed)], "batch_matched": matched})
                        break
                    else:
                        manifest["entries"].append({"tag": tag, "seed": seed, "key": key, "source": None})
        # replay check: a paired run that re-evaluated a true condition must reproduce the original bits exactly
        for label, stem, batch in candidates(tag, seed, "cf"):
            r2, b2 = load(stem)
            if r2 is None:
                continue
            for k in b2:
                if k in bits and not any(c in k for c in CF + ENT) and k != "1.0|dense":
                    manifest["replay_checks"].append({"tag": tag, "seed": seed, "source": label, "key": k, "batch": batch,
                                                      "identical": b2[k]["test_q"] == bits[k]["test_q"]})
        with open(OUT / f"{tag}{suf}.jsonl", "w") as f:
            for r in recs.values():
                f.write(json.dumps(r) + "\n")
        (OUT / f"{tag}{suf}_bits.json").write_text(json.dumps(bits))

# theory rows: counterfactual selections from the paired measurement
for tag in ("qwen15b", "olmo1b", "olmo7b"):
    o = [json.loads(l) for l in open(ORIG / f"{tag}_theory.jsonl")]
    a = [json.loads(l) for l in open(PR / "a3_theory" / f"{tag}_aligned_theory.jsonl")]
    swap = {"actnorm:MIX:KG-D~", "actnorm:MIX:KG-QA~"}
    # rows that do not involve the counterfactual files must agree between the two measurements
    idx = {(r["target"], r["layer"], r["sel"]): r for r in a}
    worst = 0.0
    for r in o:
        k = (r["target"], r["layer"], r["sel"])
        if r["sel"] in swap or k not in idx:
            continue
        for fmt in ("question", "cloze"):
            if r.get(fmt) and idx[k].get(fmt):
                for c, x in r[fmt].items():
                    if isinstance(x, dict) and c in idx[k][fmt]:
                        worst = max(worst, abs(x["ratio"] - idx[k][fmt][c]["ratio"]))
    merged = [r for r in o if r["sel"] not in swap] + [r for r in a if r["sel"] in swap]
    with open(OUT / f"{tag}_theory.jsonl", "w") as f:
        for r in merged:
            f.write(json.dumps(r) + "\n")
    manifest["replay_checks"].append({"tag": tag, "theory": True, "max_ratio_diff_shared_rows": worst})

(OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
e = manifest["entries"]
print(f"replaced {sum(1 for x in e if x['source'])} entries, missing {sum(1 for x in e if not x['source'])}, "
      f"batch-matched {sum(1 for x in e if x.get('batch_matched'))}, not batch-matched "
      f"{sorted({(x['tag'], x['seed']) for x in e if x['source'] and not x['batch_matched']})}")
rc = [c for c in manifest["replay_checks"] if "identical" in c]
print(f"replay checks: {sum(c['identical'] for c in rc)}/{len(rc)} identical; "
      f"non-identical: {sorted({(c['tag'], c['seed'], c['source'], c['batch']) for c in rc if not c['identical']})}")
print("theory shared-row max ratio difference:", [round(c['max_ratio_diff_shared_rows'], 12) for c in manifest['replay_checks'] if c.get('theory')])
