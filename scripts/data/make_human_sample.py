"""A blind sample of answers for a human check of the scoring rules.

Usage:  python scripts/data/make_human_sample.py      (after scripts/experiments/judge.py)
Draws 120 dense-model answers (question queries) from Qwen2.5 and OLMo-2, seed 0: 40 where the alias rule says correct
and the Phi-4 judge says wrong, 40 the other way round, 20 where only the strict rules (word or exact) disagree with
the alias rule, and 20 where all rules agree (a control). Writes
- results/answers/human_check.csv: id, question, accepted answers, model answer, and an empty column "correct (y/n)".
  No rule decision is shown.
- results/answers/human_check_key.json: the rule decisions for each id (to be compared with the human labels).
"""
import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from kgcal import paths as P            # noqa: E402
from kgcal import data as D            # noqa: E402

TAGS = ["qwen15b", "olmo1b", "qwen7b", "olmo7b"]
OUT = P.ANSWERS
rows = D.load_popqa()
TEST = D.facts_by_split(rows, D.make_split(rows))["test"]
pool = {"alias_only": [], "judge_only": [], "strict_only": [], "agree": []}
for tag in TAGS:
    sc = json.loads((OUT / f"{tag}_scores.json").read_text())
    ans = json.loads((P.PAIRED_CLOZE / f"{tag}_pc_answers.json").read_text())["1.0|dense"]["test_q"]
    R = {r: v["1.0|dense"]["test_q"] for r, v in sc["rules"].items()}
    assert "judge" in R and "judge_llama" in R, f"{tag}: Phi-4 judge missing"
    for i in range(len(TEST)):
        d = {r: R[r][i] == "1" for r in R}
        item = (tag, i, ans[i].strip().split("\n")[0], d)
        if d["alias"] and not d["judge"]:
            pool["alias_only"].append(item)
        elif d["judge"] and not d["alias"]:
            pool["judge_only"].append(item)
        elif (d["word"] != d["alias"]) or (d["exact"] != d["alias"]):
            pool["strict_only"].append(item)
        elif d["alias"]:
            pool["agree"].append(item)
rng = random.Random(0)
take = {"alias_only": 40, "judge_only": 40, "strict_only": 20, "agree": 20}
sample = [(k, it) for k, n in take.items() for it in rng.sample(pool[k], min(n, len(pool[k])))]
rng.shuffle(sample)
key = {}
with open(OUT / "human_check.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "question", "accepted answers", "model answer", "correct (y/n)"])
    for n, (k, (tag, i, a, d)) in enumerate(sample, 1):
        w.writerow([n, TEST[i]["question"], "; ".join(D.ans(TEST[i])), a, ""])
        key[n] = {"stratum": k, "tag": tag, "fact": i, "rules": d}
(OUT / "human_check_key.json").write_text(json.dumps({"pool_sizes": {k: len(v) for k, v in pool.items()}, "items": key},
                                                     indent=1))
print("pool sizes:", {k: len(v) for k, v in pool.items()}, "| sample:", len(sample), "items ->", OUT / "human_check.csv")
