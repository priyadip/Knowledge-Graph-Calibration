"""EntityQuestions test sample for the transfer test.

Usage:  python scripts/data/build_entityquestions.py      (after scripts/data/download_data.py)
- Source: data/entityquestions/raw/dataset.zip (Sciavolino et al., 2021; nlp.cs.princeton.edu/projects/entity-questions,
  sha256 recorded in the output). Test split only.
- Relations: the 18 not among PopQA's 16 (principal set, up to 222 questions each; P800 has 220, so 3,994 in all) and
  the 6 shared ones
  (P17, P19, P36, P50, P106, P136; 100 each, reported separately).
- The subject of a question is the text between the relation template's fixed prefix and suffix (longest common
  prefix and suffix of the relation's test questions). Questions whose subject is a subject of the PopQA calibration
  facts (the entities whose statements are in the calibration text) are removed before sampling.
- Sampling: seed 0, fixed before any model sees the questions. Facts are written in the format kgcal.evaluate expects
  ({"question", "possible_answers"}), plus relation and subject.
"""
import hashlib
import json
import os
import random
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from kgcal import paths as P            # noqa: E402
from kgcal import data as D            # noqa: E402

RAW = P.EQ / "raw" / "dataset.zip"
OUT = P.EQ / "eq_test_sample.json"
SHARED = {"P17", "P19", "P36", "P50", "P106", "P136"}
assert SHARED == {p for p in {v[0] for v in D.REL.values()}} & SHARED
N_MAIN, N_SHARED = 222, 100


def template(qs):
    pre = os.path.commonprefix(qs)
    suf = os.path.commonprefix([q[::-1] for q in qs])[::-1]
    # keep whole words: the subject never starts or ends inside the fixed text
    pre = pre[:pre.rfind(" ") + 1] if not pre.endswith(" ") else pre
    suf = suf[suf.find(" "):] if suf and not suf.startswith(" ") and " " in suf else suf
    return pre, suf


rows = D.load_popqa()
F = D.facts_by_split(rows, D.make_split(rows))
cal_subj = {r["subj"].lower() for r in F["cal"]}
z = zipfile.ZipFile(RAW)
names = sorted(n for n in z.namelist() if n.startswith("dataset/test/") and n.endswith(".test.json"))
rng = random.Random(0)
out, info = [], {}
for n in names:
    prop = n.rsplit("/", 1)[1].split(".")[0]
    data = json.loads(z.read(n))
    pre, suf = template([d["question"] for d in data])
    facts = []
    for d in data:
        q = d["question"]
        assert q.startswith(pre) and q.endswith(suf), (prop, q)
        subj = q[len(pre):len(q) - len(suf)] if suf else q[len(pre):]
        if prop == "P40" and subj.endswith("'s"):        # "Who is X's child?"
            subj = subj[:-2]
        facts.append({"question": q, "possible_answers": json.dumps(d["answers"]), "eq_prop": prop, "subj": subj,
                      "shared_with_popqa": prop in SHARED})
    kept = [f for f in facts if f["subj"].lower() not in cal_subj]
    k = N_SHARED if prop in SHARED else N_MAIN
    pick = rng.sample(kept, min(k, len(kept)))
    out += pick
    info[prop] = {"template": [pre, suf], "test": len(facts), "removed_cal_subject": len(facts) - len(kept),
                  "sampled": len(pick)}
main = [f for f in out if not f["shared_with_popqa"]]
OUT.write_text(json.dumps({"source": str(RAW.relative_to(P.ROOT)), "sha256": hashlib.sha256(RAW.read_bytes()).hexdigest(),
                           "relations": info, "n_main": len(main), "n_shared": len(out) - len(main), "facts": out},
                          indent=1))
for p, v in info.items():
    print(p, v)
print("main", len(main), "shared", len(out) - len(main))
