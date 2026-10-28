"""Two analyses of existing results: name overlap of EntityQuestions with the PopQA calibration text, and format
against the number of covered facts.

Usage:  python scripts/analysis/overlap_coverage.py      (before scripts/paper/make_numbers.py)
1. EntityQuestions overlap with the PopQA calibration text. For each of the 3,994 test questions of the 18
   relations outside PopQA: does its subject occur as a whole word in the PopQA calibration text that the transfer test
   uses (KG-D and KG-D~ lines), does its subject's name equal the name of a PopQA calibration subject or an answer alias
   of a PopQA calibration fact, and does one of its answers occur in that text? All three are name matches: the dataset
   gives no Wikidata identifiers, so entity identity cannot be checked. Questions whose subject name equals a PopQA
   calibration subject were already removed when the sample was drawn (build_entityquestions.py). Counts are given per question and
   per unique subject. Three subsets are analysed, each with its own rule: subject-disjoint (neither subject criterion),
   answer-disjoint (no answer occurs in the calibration text), and both (neither kind of overlap). Both channels on MIX(KG-D) minus C4/C4 is recomputed on each subset with the
   paired bootstrap of the paper (10,000 draws, seed 0).
2. Format and the number of covered facts. The KG half of MIX(KG-D) and MIX(KG-QA) is packed as in the main
   runs (Qwen2.5 tokenizer, 64 sequences of 512 tokens); we count the facts in each and compare their relation mixes.
   On the four Qwen2.5 and OLMo-2 models, the share sweep gives the gain of both channels with 32 and 64 sequences of
   KG-D; linear interpolation to the number of question-answer facts bounds how much of the format contrast the
   difference in covered facts could explain (approximate, since the sweep also changes the C4 share).
Writes outputs/tables/overlap.json.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from kgcal import paths as P            # noqa: E402
from kgcal import data as D                     # noqa: E402

MODELS = ["qwen15b", "olmo1b", "qwen7b", "olmo7b", "llama8b", "mistral7b", "phi4"]
PAPER = MODELS[:4]
BUD = ("0.7", "0.5")
OUT = {}


def paired(a, b, m, n=10000):
    x, y = a[m].astype(float), b[m].astype(float)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    d = (y[idx] - x[idx]).mean(1)
    lo, hi = np.quantile(d, [0.025, 0.975])
    return {"diff": float(100 * (y.mean() - x.mean())), "lo": float(100 * lo), "hi": float(100 * hi),
            "sig": bool(lo > 0 or hi < 0), "n": int(m.sum())}


# ---------------------------------------------------------------- 1. EntityQuestions overlap
facts = json.loads((P.EQ / "eq_test_sample.json").read_text())["facts"]
main = np.array([not f["shared_with_popqa"] for f in facts])
lines = [json.loads(l) for src in (P.CALIB / "KG-D.jsonl", P.calib_file("KG-D~"))
         for l in open(src)]
text = "\n".join(lines).lower()
rows = D.load_popqa()
F = D.facts_by_split(rows, D.make_split(rows))
cal_subjects = {r["subj"].lower() for r in F["cal"]}
cal_answers = {a.lower() for r in F["cal"] for a in D.ans(r)}
cal_entities = cal_subjects | cal_answers


def occurs(s):
    s = s.lower().strip()
    return len(s) >= 3 and re.search(r"(?<![a-z0-9])" + re.escape(s) + r"(?![a-z0-9])", text) is not None


subj = [f["subj"] for f in facts]
in_text = np.array([occurs(s) for s in subj])
is_entity = np.array([s.lower() in cal_entities for s in subj])
ans_in_text = np.array([any(occurs(a) for a in json.loads(f["possible_answers"])) for f in facts])
union = in_text | is_entity
uniq = lambda m: len({subj[i].lower() for i in np.flatnonzero(m & main)})
OUT["overlap"] = {
    "questions": int(main.sum()), "unique_subjects": uniq(np.ones(len(facts), bool)),
    "subject_in_text": {"questions": int((in_text & main).sum()), "unique": uniq(in_text)},
    "subject_is_calibration_entity": {"questions": int((is_entity & main).sum()), "unique": uniq(is_entity),
                                      "as_subject_name": int(sum(s.lower() in cal_subjects for s, m in zip(subj, main) if m)),
                                      "as_answer_alias": int(sum(s.lower() in cal_answers for s, m in zip(subj, main) if m))},
    "subject_either": {"questions": int((union & main).sum()), "unique": uniq(union)},
    "answer_in_text": {"questions": int((ans_in_text & main).sum())},
    "subject_disjoint_questions": int((main & ~union).sum()),
    "answer_disjoint_questions": int((main & ~ans_in_text).sum()),
    "both_disjoint_questions": int((main & ~union & ~ans_in_text).sum()),
}
sets = {"all": main, "subject_disjoint": main & ~union, "answer_disjoint": main & ~ans_in_text,
        "both_disjoint": main & ~union & ~ans_in_text}
res = {}
for tag in MODELS:
    b = json.loads((P.EQ_RESULTS / f"{tag}_eq_bits.json").read_text())
    B = lambda k: np.array([c == "1" for c in b[k]["eq"]])
    dense = B("1.0|dense")
    for t in BUD:
        for name, m in sets.items():
            res[f"{tag}|{t}|{name}"] = paired(B(f"{t}|C4|C4"), B(f"{t}|MIX:KG-D|MIX:KG-D"), m & dense)
OUT["eq_subsets"] = res

# ---------------------------------------------------------------- 2. format and the number of covered facts
from transformers import AutoTokenizer          # noqa: E402
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
d2p = {D.declarative(r["prop"], r["subj"], D.ans(r)[0]): r["prop"] for r in F["cal"]}
q2p = {f"Q: {r['question']} A: {D.ans(r)[0]}": r["prop"] for r in F["cal"]}


def covered(path, mapping, n_tok=64 * 512):
    ls = [json.loads(l) for l in open(path)]
    lens = np.cumsum([len(tok(t + "\n", add_special_tokens=False)["input_ids"]) for t in ls])
    k = int(np.searchsorted(lens, n_tok))
    return Counter(mapping.get(t, "?") for t in ls[:k]), k


cd, nd = covered(P.CALIB / "KG-D.jsonl", d2p)
cq, nq = covered(P.CALIB / "KG-QA.jsonl", q2p)
props = sorted(set(cd) | set(cq))
tv = 0.5 * sum(abs(cd[p] / sum(cd.values()) - cq[p] / sum(cq.values())) for p in props)
S = json.loads((P.TABLES / "summary.json").read_text())
bound = {}
for tag in PAPER:
    mb = json.loads((P.MIXTURE / f"{tag}_mix_bits.json").read_text())
    pb = json.loads((P.MERGED / f"{tag}_bits.json").read_text())
    m = np.array([c == "1" for c in pb["1.0|dense"]["test_q"]])
    R = lambda d, k: 100 * np.array([c == "1" for c in d[k]["test_q"]])[m].mean()
    for t in BUD:
        g25 = R(mb, f"{t}|R32|R32") - R(mb, f"{t}|R0|R0")
        g50 = R(pb, f"{t}|MIX:KG-D|MIX:KG-D") - R(mb, f"{t}|R0|R0")
        at_nq = g25 + (g50 - g25) * (nq - nd / 2) / (nd / 2)
        bound[f"{tag}|{t}"] = {"gain_25": g25, "gain_50": g50, "gain_at_qa_facts": at_nq, "coverage_bound": g50 - at_nq,
                               "format_contrast": 100 * S[tag]["settings"][t]["contrasts"]["Format"]["diff"]}
OUT["format_coverage"] = {"facts_decl": int(nd), "facts_qa": int(nq), "relation_tv": float(tv), "bound": bound}
(P.TABLES / "overlap.json").write_text(json.dumps(OUT, indent=1))
print(json.dumps(OUT["overlap"], indent=1))
print({k: round(v["coverage_bound"], 1) for k, v in bound.items()})
