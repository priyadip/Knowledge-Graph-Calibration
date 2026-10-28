"""Paired variants of the calibration texts: only the order of lines changes.

python scripts/data/build_paired_variants.py      ->  data/calib_paired/KG-D~.jsonl, KG-QA~.jsonl, KG-S.jsonl

- KG-D~: line i is the counterfactual of the fact on line i of KG-D (same subject and relation, the same wrong object
  as in data/calib/KG-D~.jsonl). KG-QA~ likewise against KG-QA.
- KG-S: line i has the relation and the true object of line i of KG-D; only the subject name differs (the made-up
  names are the lines of data/calib/KG-S.jsonl, reordered).
Each paired file holds exactly the lines of its unpaired file, so a true and a variant calibration text built from the
same lines cover the same facts.
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from kgcal import data as D  # noqa: E402
from kgcal import paths as P  # noqa: E402

rows = D.load_popqa()
F = D.facts_by_split(rows, D.make_split(rows))
prefixes = {D.cloze(r["prop"], r["subj"]) for r in F["cal"]}
TPL = [(rel, t.split("{s}")[0], t.split("{s}")[1].split("{o}")[0], t.split("{o}")[1]) for rel, (_, t) in D.REL.items()]


def lines(name):
    return [json.loads(l) for l in open(P.CALIB / f"{name}.jsonl")]


def decl_key(s):
    """Subject-relation prefix of a declarative statement (the cloze text), longest match."""
    w = s.rstrip(".").split(" ")
    for i in range(len(w) - 1, 0, -1):
        p = " ".join(w[:i])
        if p in prefixes:
            return p
    raise KeyError(s)


def qa_key(s):
    return s.split(" A: ")[0]


def rel_obj_key(s):
    """(relation, object) of a templated statement; the subject is whatever stands between prefix and middle."""
    hits = []
    for rel, pre, mid, post in TPL:
        if s.startswith(pre) and s.endswith(post) and mid in s[len(pre):]:
            body = s[len(pre):len(s) - len(post)]
            hits.append((len(pre) + len(mid) + len(post), rel, body.rsplit(mid, 1)[1]))
    if not hits:
        raise KeyError(s)
    _, rel, obj = max(hits)                                       # most specific template wins
    return rel, obj


def align(true_name, var_name, key):
    true, var = lines(true_name), lines(var_name)
    if key is rel_obj_key:
        assert Counter(map(key, true)) == Counter(map(key, var)), "(relation, object) multisets differ"
    bucket = defaultdict(list)
    for s in var:
        bucket[key(s)].append(s)
    out = [bucket[key(s)].pop() for s in true]
    assert all(len(v) == 0 for v in bucket.values()), "unmatched variant lines"
    assert sorted(out) == sorted(var), "a paired file must contain exactly the lines of its unpaired file"
    with open(P.CALIB_PAIRED / f"{var_name}.jsonl", "w") as f:
        for s in out:
            f.write(json.dumps(s) + "\n")
    same = sum(key(a) == key(b) for a, b in zip(true, out))
    print(f"{var_name}: {len(out)} lines, matched to {true_name} line by line: {same}/{len(out)}")


P.CALIB_PAIRED.mkdir(parents=True, exist_ok=True)
align("KG-D", "KG-D~", decl_key)
align("KG-QA", "KG-QA~", qa_key)
align("KG-D", "KG-S", rel_obj_key)
