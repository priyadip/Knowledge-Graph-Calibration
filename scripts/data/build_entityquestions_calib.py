"""Calibration text for the EntityQuestions replication.

Usage:  python scripts/data/build_entityquestions_calib.py      (after build_entityquestions.py)
- Calibration facts: EntityQuestions train triples (Wikidata, via T-REx) of the 18 relations outside PopQA, 400 per
  relation (P800 has fewer), seed 0, after removing every fact whose subject occurs in the test sample, so the test
  subjects are unseen. These play the role of the PopQA calibration facts of the paper.
- EQ-D: one declarative statement per fact (templates below, written in the style of kgcal.data.REL).
  EQ-D~: the same line with the object replaced by an object of the same relation from the calibration pool that is
  not a gold answer of the fact (no type check: the dataset carries no Wikidata IDs).
  EQ-QA and EQ-QA~: "Q: <question> A: <answer>" with the true and the same counterfactual object.
  Paired files are written line by line in one shared order, so a true and a counterfactual mixture cover the same
  facts, as in data/calib_paired.
Writes data/calib_eq/<name>.jsonl and MANIFEST.json.
"""
import hashlib
import json
import random
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from kgcal import paths as P            # noqa: E402
RAW = P.EQ / "raw" / "dataset.zip"
SAMPLE = json.loads((P.EQ / "eq_test_sample.json").read_text())
OUT = P.CALIB_EQ
PER_REL = 400
TEMPLATES = {
    "P112": "{s} was founded by {o}.", "P127": "{s} is owned by {o}.", "P131": "{s} is located in {o}.",
    "P159": "The headquarters of {s} is in {o}.", "P170": "{s} was created by {o}.", "P175": "{s} was performed by {o}.",
    "P176": "{s} is produced by {o}.", "P20": "{s} died in {o}.", "P26": "{s} is married to {o}.",
    "P264": "{s} is represented by the music label {o}.", "P276": "{s} is located in {o}.", "P40": "{s}'s child is {o}.",
    "P407": "{s} was written in {o}.", "P413": "{s} plays the position of {o}.", "P495": "{s} was created in {o}.",
    "P69": "{s} was educated at {o}.", "P740": "{s} was founded in {o}.", "P800": "{s} is famous for {o}.",
}

test_subj = {f["subj"].lower() for f in SAMPLE["facts"]}
z = zipfile.ZipFile(RAW)
rng = random.Random(0)
facts = []
for p in sorted(TEMPLATES):
    pre, suf = SAMPLE["relations"][p]["template"]
    data = json.loads(z.read(f"dataset/train/{p}.train.json"))
    kept = []
    for d in data:
        q = d["question"]
        s = q[len(pre):len(q) - len(suf)] if suf else q[len(pre):]
        if p == "P40" and s.endswith("'s"):
            s = s[:-2]
        if s.lower() not in test_subj:
            kept.append({"prop": p, "subj": s, "question": q, "answers": d["answers"]})
    facts += rng.sample(kept, min(PER_REL, len(kept)))
pool = {}
for f in facts:
    pool.setdefault(f["prop"], []).append(f["answers"][0])
for f in facts:
    gold = {a.lower() for a in f["answers"]}
    f["cf"] = rng.choice([o for o in pool[f["prop"]] if o.lower() not in gold])
rng.shuffle(facts)
src = {"EQ-D": [TEMPLATES[f["prop"]].format(s=f["subj"], o=f["answers"][0]) for f in facts],
       "EQ-D~": [TEMPLATES[f["prop"]].format(s=f["subj"], o=f["cf"]) for f in facts],
       "EQ-QA": [f"Q: {f['question']} A: {f['answers'][0]}" for f in facts],
       "EQ-QA~": [f"Q: {f['question']} A: {f['cf']}" for f in facts]}
OUT.mkdir(parents=True, exist_ok=True)
man = {"source": str(RAW.relative_to(P.ROOT)), "sha256": hashlib.sha256(RAW.read_bytes()).hexdigest(), "facts": len(facts),
       "per_relation": PER_REL, "templates": TEMPLATES, "files": {}}
for k, lines in src.items():
    with open(OUT / f"{k}.jsonl", "w") as fh:
        for line in lines:
            fh.write(json.dumps(line) + "\n")
    man["files"][k] = {"lines": len(lines), "sha256": hashlib.sha256((OUT / f"{k}.jsonl").read_bytes()).hexdigest()}
(OUT / "MANIFEST.json").write_text(json.dumps(man, indent=1))
for k in src:
    print(k, src[k][:2])
print(len(facts), "facts")
