"""PopQA split, Wikidata and Wikipedia caches, calibration texts and leakage checks, in the order they were built.

KGCAL_USER_AGENT="<your contact>" python scripts/data/build_popqa_kg.py

1. Subject-disjoint PopQA split (seed 0) over the 12,244 PopQA subjects: data/popqa/popqa_split.json.
2. Wikidata statements of every PopQA subject, the labels of their properties and the types (P31) of the answer
   entities, and the Wikipedia lead paragraphs of the calibration subjects: data/wikidata/*.json. Only items missing
   from these caches are fetched, so with the released caches nothing is queried.
3. Calibration texts (data/calib): KG-D, KG-D~, KG-QA, KG-QA~, KG-S, WIKI, KG-O and KG-A from
   kgcal.data.build_calibration_sources, then KG-O2 (statements of the other relations) and KG-Ac
   (kgcal.data.build_anchor_clean).
4. Leakage checks: data/popqa/leakage.json, leakage_strict.json and leakage_flags.json.
The paper reports KG-O2 and KG-Ac. KG-O and KG-A are earlier sets of the same statements that only the Qwen2.5-1.5B
and OLMo-2-1B grids scored; they were written before generic_statement handled property labels that are verb
phrases, so the KG-O and KG-A written here differ from the released files, and the released files are the ones those
grids read. Every other file is rewritten identically from the released caches (scripts/data/check_data.py).
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from kgcal import data as D  # noqa: E402

for d in (D.SPLITS, D.KG, D.CALIB):
    d.mkdir(parents=True, exist_ok=True)
if not (D.SPLITS / "popqa_split.json").exists():
    # the split is drawn over all 12,244 PopQA subjects, including the three of the rows without a subject name that
    # load_popqa drops; make_split uses only the subject identifiers and writes the file
    from datasets import load_dataset
    D.make_split([dict(r) for r in load_dataset("akariasai/PopQA", split="test",
                                                revision=D.HF_REVISIONS["akariasai/PopQA"])])
rows = D.load_popqa()
split = D.make_split(rows)
print({k: len(v) for k, v in split.items() if k != "seed"})
F = D.facts_by_split(rows, split)

st = D.fetch_statements(sorted({D.qid(r["s_uri"]) for r in rows}))
D.fetch_property_labels(sorted({p for v in st.values() for p, _, _ in v}))
D.fetch_types(sorted({D.qid(r["o_uri"]) for r in rows}))
D.fetch_wiki_summaries(sorted({r["s_wiki_title"] for r in F["cal"]}))

src = D.build_calibration_sources(rows, split)
print({k: len(v) for k, v in src.items()})
print("leakage", D.leakage_report(rows, split, src))

plab = json.loads((D.KG / "property_labels.json").read_text())
rng = random.Random(0)
other = [D.generic_statement(r["subj"], plab[p], ol) for r in F["cal"]
         for p, oq, ol in st.get(D.qid(r["s_uri"]), [])
         if p not in D.TESTED_PROPS and p not in D.SKIP_PROPS and p in plab]
rng.shuffle(other)
with open(D.CALIB / "KG-O2.jsonl", "w") as f:
    for line in other:
        f.write(json.dumps(line) + "\n")
print("KG-O2", len(other), "KG-Ac", len(D.build_anchor_clean(rows, split)))
print("leakage_strict", D.leakage_strict(rows, split))
print("leakage_flags", D.leakage_flags(rows, split))
