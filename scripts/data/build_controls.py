"""Data for the controls for format, scoring and truth, written once before any run.

Usage:  python scripts/data/build_controls.py
- data/controls/fmt_facts.json: the facts of the paper's KG-QA source in its order, each with its question-answer line
  and its declarative line (same fact, same relation template as KG-D).
- data/controls/truth_variants.json: line-by-line variants of the paper's KG-D source (same facts, same order): V25,
  V50, V75 (nested random shares of counterfactual lines from the paired KG-D~ file, seed 0), VX (implausible objects:
  another relation, no shared Wikidata type, same token count within one under the Qwen2.5 tokenizer, seed 0) and VC
  (each fact twice, true and counterfactual in random order, seed 0).
- data/controls/test_mc.json: four candidates per test fact: the first gold alias and three answers of other PopQA facts
  of the same relation that neither equal nor contain nor are contained in a gold alias (seed 0).
"""
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from kgcal import paths as P            # noqa: E402
from kgcal import data as D                     # noqa: E402
from transformers import AutoTokenizer          # noqa: E402

OUT = P.CONTROLS
OUT.mkdir(parents=True, exist_ok=True)
rows = D.load_popqa()
F = D.facts_by_split(rows, D.make_split(rows))
cal, test = F["cal"], F["test"]
types = json.loads((P.KG / "types.json").read_text())
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
ntok = lambda s: len(tok(" " + s, add_special_tokens=False)["input_ids"])
read = lambda p: [json.loads(l) for l in open(p)]
man = {}

# ---------------------------------------------------------------- matched format
qa_lines = read(P.CALIB / "KG-QA.jsonl")
by_qa = {f"Q: {r['question']} A: {D.ans(r)[0]}": r for r in cal}
fmt = []
for line in qa_lines:
    r = by_qa[line]
    fmt.append({"qa": line, "decl": D.declarative(r["prop"], r["subj"], D.ans(r)[0]), "prop": r["prop"]})
(OUT / "fmt_facts.json").write_text(json.dumps(fmt))
man["fmt_facts"] = len(fmt)

# ---------------------------------------------------------------- truth variants
true = read(P.CALIB / "KG-D.jsonl")
cf = read(P.calib_file("KG-D~"))
assert len(true) == len(cf)
by_decl = {D.declarative(r["prop"], r["subj"], D.ans(r)[0]): r for r in cal}
facts = [by_decl[l] for l in true]
n = len(true)
perm = list(range(n))
random.Random(0).shuffle(perm)
rank = {i: k for k, i in enumerate(perm)}
variants = {}
for share in (25, 50, 75):
    variants[f"V{share}"] = [cf[i] if rank[i] < share * n // 100 else true[i] for i in range(n)]
# implausible objects
pool = []
seen = set()
for r in rows:
    o, q = D.ans(r)[0], D.qid(r["o_uri"])
    if (r["prop"], o) not in seen:
        seen.add((r["prop"], o))
        pool.append((r["prop"], o, q, ntok(o), set(types.get(q, []))))
rng = random.Random(0)
vx, typed_ok, widened = [], 0, {}
for r in facts:
    o, q = D.ans(r)[0], D.qid(r["o_uri"])
    gold = {a.lower() for a in D.ans(r)}
    t0, n0 = set(types.get(q, [])), ntok(o)
    for tol in (1, 2, 3, 5, 8, 1000):              # widen the length tolerance only where no candidate exists
        cands = [c for c in pool if c[0] != r["prop"] and abs(c[3] - n0) <= tol and c[1].lower() not in gold
                 and not (t0 & c[4])]
        if cands:
            break
    widened[tol] = widened.get(tol, 0) + 1
    choice = rng.choice(cands)
    typed_ok += bool(t0) and bool(choice[4])
    vx.append(D.declarative(r["prop"], r["subj"], choice[1]))
variants["VX"] = vx
rng = random.Random(0)
vc = []
for i in range(n):
    pair = [true[i], cf[i]]
    rng.shuffle(pair)
    vc.append(pair)
variants["VC_pairs"] = vc
(OUT / "truth_variants.json").write_text(json.dumps(variants))
man["truth"] = {"lines": n, "vx_type_checked": typed_ok, "vx_fallback_no_types": n - typed_ok,
                "vx_length_tolerance_used": {str(k): v for k, v in sorted(widened.items())}}

# ---------------------------------------------------------------- multiple choice
answers_by_prop = {}
for r in rows:
    answers_by_prop.setdefault(r["prop"], [])
    a = D.ans(r)[0]
    if a not in answers_by_prop[r["prop"]]:
        answers_by_prop[r["prop"]].append(a)
rng = random.Random(0)
mc, short = [], 0
for i, r in enumerate(test):
    gold = [a.lower() for a in D.ans(r)]
    ok = [a for a in answers_by_prop[r["prop"]]
          if not any(a.lower() == g or a.lower() in g or g in a.lower() for g in gold)]
    k = min(3, len(ok))
    short += k < 3
    mc.append({"i": i, "cands": [D.ans(r)[0]] + rng.sample(ok, k)})
(OUT / "test_mc.json").write_text(json.dumps(mc))
man["mc"] = {"items": len(mc), "items_with_fewer_than_3_distractors": short}
for f in ("fmt_facts.json", "truth_variants.json", "test_mc.json"):
    man[f + ".sha256"] = hashlib.sha256((OUT / f).read_bytes()).hexdigest()
(OUT / "MANIFEST.json").write_text(json.dumps(man, indent=1))
print(json.dumps(man, indent=1))
print("examples VX:", variants["VX"][:3])
print("examples MC:", mc[:2])
