# Inputs

Every file here is an input of the runs, exactly as the runs read it. `python scripts/data/check_data.py` compares the
files with the SHA-256 hashes of `MANIFEST.json`. The builders are in `scripts/data/`; with the released Wikidata and
Wikipedia caches they write the same files again, except `calib/KG-O.jsonl` and `calib/KG-A.jsonl` (see below).
Calibration files hold one JSON string per line.

## popqa/ (`scripts/data/build_popqa_kg.py`)

| File | Content |
|---|---|
| `popqa_split.json` | subject-disjoint split of the PopQA test set (seed 0) over its 12,244 subjects: 6,122 calibration, 1,224 validation and 4,898 test subjects (Wikidata identifiers). Three PopQA rows have no subject name and are dropped when the facts are loaded, so 6,121 calibration and 4,896 test subjects have facts: 7,139 calibration, 1,435 validation and 5,690 test facts. |
| `leakage.json` | share of test facts whose answer string occurs in each calibration text |
| `leakage_strict.json`, `leakage_flags.json` | fact-level checks: homonyms of calibration subjects, loose answer matches in the anchored statements, subject and answer together in a Wikipedia paragraph (one flag per test fact, in test-fact order) |

## wikidata/ (`scripts/data/build_popqa_kg.py`, retrieved October 2026)

| File | Content |
|---|---|
| `statements.json` | item-valued statements of every PopQA subject: subject identifier to a list of [property, object identifier, object label] |
| `property_labels.json` | English labels of those properties |
| `types.json` | instance-of (P31) types of the PopQA answer entities, for type-consistent counterfactual objects |
| `wiki_summaries.json` | Wikipedia lead paragraph of every calibration subject |

## calib/ (`scripts/data/build_popqa_kg.py`)

All texts are written from the 7,139 facts of the calibration subjects, or from Wikidata statements; no test fact
occurs in them.

| File | Lines | Content |
|---|---|---|
| `KG-D.jsonl` | 7,139 | one declarative statement per calibration fact (relation templates of `kgcal/data.py`) |
| `KG-QA.jsonl` | 7,139 | the same facts as question and answer ("Q: ... A: ...") |
| `KG-D~.jsonl`, `KG-QA~.jsonl` | 7,139 | the same subjects and relations with a wrong object of the same relation, of the gold object's type where one exists |
| `KG-S.jsonl` | 7,139 | the true relations and objects with made-up subject names |
| `KG-O2.jsonl` | 156,065 | statements of the calibration subjects about the other Wikidata relations (none of the 16 tested ones) |
| `KG-Ac.jsonl` | 130,436 | statements of held-out subjects without the tested relation and without any object that matches an answer alias |
| `WIKI.jsonl` | 6,463 | Wikipedia lead paragraphs of the calibration subjects |
| `KG-O.jsonl`, `KG-A.jsonl` | 156,065, 173,665 | earlier sets of the KG-O2 and KG-Ac statements that only the Qwen2.5-1.5B and OLMo-2-1B grids scored; the paper does not report them. They were written before the verbalizer handled property labels that are verb phrases, and KG-A removed exact answer matches only, so the builder now writes them differently. |

The variants in `calib/` were shuffled independently of their true texts; the runs of `results/grid` used them.
Every result the paper reports for KG-D~, KG-QA~ and KG-S uses the paired files below.

## calib_paired/ (`scripts/data/build_paired_variants.py`)

`KG-D~.jsonl` and `KG-QA~.jsonl` reordered so that line i is the counterfactual of the fact on line i of `KG-D.jsonl`
or `KG-QA.jsonl`, and `KG-S.jsonl` reordered so that line i has the relation and object of line i of `KG-D.jsonl`.
Each holds exactly the lines of its file in `calib/`, so a true text and its variant built from the same line numbers
cover the same facts.

## entityquestions/ (`scripts/data/build_entityquestions.py`)

`eq_test_sample.json`: 3,994 test questions of the 18 EntityQuestions relations outside PopQA (222 per relation,
220 for P800) and 600 of the 6 shared relations (100 each), seed 0, without questions whose subject is a PopQA
calibration subject. `raw/dataset.zip` is the downloaded dataset (`scripts/data/download_data.py`); it is not
released here.

## calib_eq/ (`scripts/data/build_entityquestions_calib.py`)

Calibration texts written from the EntityQuestions training triples of the 18 relations (400 per relation, none
about a test subject): declarative (`EQ-D`) and question-answer (`EQ-QA`) lines, and their counterfactual variants
with an object of the same relation (`EQ-D~`, `EQ-QA~`), paired line by line. `MANIFEST.json` lists the templates
and hashes.

## controls/ (`scripts/data/build_controls.py`)

| File | Content |
|---|---|
| `fmt_facts.json` | the facts of KG-QA in its order, each with its question-answer and declarative line (matched format) |
| `truth_variants.json` | variants of KG-D with the same facts in the same order: 25%, 50% and 75% counterfactual lines (nested, seed 0), implausible objects of another relation and type (VX), and each fact twice, true and counterfactual (VC) |
| `test_mc.json` | four candidates per test fact for likelihood multiple choice: the first answer alias and three answers of other facts of the same relation that do not match an alias |
