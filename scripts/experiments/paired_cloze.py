"""Paired variants scored with cloze queries too, and the generated answers of the key conditions.

Usage:  python scripts/experiments/paired_cloze.py <tag>
- Conditions (calibration seed 0, 70% and 50%): none|C4, C4|C4, C4|MIX:KG-D, MIX:KG-D|MIX:KG-D, MIX:KG-D~|MIX:KG-D~,
  MIX:KG-QA|MIX:KG-QA, MIX:KG-QA~|MIX:KG-QA~, MIX:KG-S|MIX:KG-S, plus the dense model. Question and cloze queries.
- Calibration files: KG-D and KG-QA from data/calib; KG-D~, KG-QA~ and KG-S from data/calib_paired, written line by
  line from them (the same files as the paired runs of results/paired).
- Generation batch of the original runs (96 for the 1B models, 64 for the 7B models), so every question-query result
  must reproduce the bits of results/merged exactly (checked in paper_models.py). The runs add the cloze scores of
  the paired variants, which the paired runs of results/paired did not compute.
- Answers: the decoded continuation of every test question and cloze query is kept (results/paired_cloze/
  <tag>_pc_answers.json), so the answers can be scored again with other rules. recall() below is kgcal.evaluate.recall
  with the answers kept; the correctness rule is unchanged.
Outputs go to results/paired_cloze.
"""
import json
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from kgcal import paths as P            # noqa: E402
from kgcal import data as D            # noqa: E402
from kgcal import run_grid as R        # noqa: E402
from kgcal.data import ans             # noqa: E402
from kgcal.evaluate import prompts     # noqa: E402

MODELS = {"qwen15b": "Qwen/Qwen2.5-1.5B", "olmo1b": "allenai/OLMo-2-0425-1B",
          "qwen7b": "Qwen/Qwen2.5-7B", "olmo7b": "allenai/OLMo-2-1124-7B"}
BATCH = {"qwen15b": 96, "olmo1b": 96, "qwen7b": 64, "olmo7b": 64}
GROUP = {"qwen15b": 4, "olmo1b": 4, "qwen7b": 1, "olmo7b": 2}
KEEP = ("none|C4", "C4|C4", "C4|MIX:KG-D", "MIX:KG-D|MIX:KG-D", "MIX:KG-D~|MIX:KG-D~", "MIX:KG-QA|MIX:KG-QA",
        "MIX:KG-QA~|MIX:KG-QA~", "MIX:KG-S|MIX:KG-S")
tag = sys.argv[1]
orig_grid = R.grid
R.grid = lambda name: [c for c in orig_grid("core") if c[0] in KEEP]
conds = [c[0] for c in R.grid("core")]
assert sorted(conds) == sorted(KEEP)

# run_grid evaluates the dense model, then every budget and condition in grid order, each with three recall calls:
# validation questions, test questions, test cloze queries. The answers are filed under that order and checked.
ORDER = ["1.0|dense"] + [f"{t}|{c}" for t in ("0.7", "0.5") for c in conds]
CALLS = [("val", "question"), ("test", "question"), ("test", "cloze")]
ANS: dict[str, dict[str, list[str]]] = {}
state = {"k": 0}
rows = D.load_popqa()
F = D.facts_by_split(rows, D.make_split(rows))
SPLIT_OF = {id(F["val"]): "val", id(F["test"]): "test"}


@torch.no_grad()
def recall(model, tok, facts, examples, fmt, batch=96, max_new=12):
    """kgcal.evaluate.recall, keeping the decoded continuation of every prompt."""
    cond, (split, fmt_expected) = ORDER[state["k"] // 3], CALLS[state["k"] % 3]
    assert (SPLIT_OF.get(id(facts)), fmt) == (split, fmt_expected), (state["k"], SPLIT_OF.get(id(facts)), fmt)
    state["k"] += 1
    dev = next(model.parameters()).device
    tok.padding_side = "left"
    ps = prompts(facts, examples, fmt)
    out, texts = [], []
    for b in range(0, len(ps), batch):
        enc = tok(ps[b:b + batch], return_tensors="pt", padding=True).to(dev)
        g = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        dec = tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for r, t in zip(facts[b:b + batch], dec):
            line = t.strip().split("\n")[0].lower()
            out.append(any(a.lower() in line for a in ans(r)))
            texts.append(t)
    if split == "test":
        ANS.setdefault(cond, {})["test_q" if fmt == "question" else "test_c"] = texts
        (out_dir / f"{new_tag}_answers.json").write_text(json.dumps(ANS))
    return out


R.D.load_popqa = lambda: rows                       # the same list objects, so the split can be recognised
_orig_facts = R.D.facts_by_split
R.D.facts_by_split = lambda rows_, split_: F
R.E.recall = recall

out_dir = P.PAIRED_CLOZE
out_dir.mkdir(parents=True, exist_ok=True)
new_tag = f"{tag}_pc"
shutil.copy(P.GRID / f"{tag}_gride.json", out_dir / f"{new_tag}_gride.json")
D.PAIRED = True
D.RESULTS = out_dir
for p in (out_dir / f"{new_tag}.jsonl", out_dir / f"{new_tag}_bits.json", out_dir / f"{new_tag}_answers.json"):
    assert not p.exists(), f"{p} exists: a resumed run would break the answer order; move it away first"
sys.argv = ["run_grid", "--model", MODELS[tag], "--tag", new_tag, "--grid", "core", "--batch", str(BATCH[tag]),
            "--group", str(GROUP[tag])]
R.main()
assert state["k"] == 3 * len(ORDER), state["k"]
print("JOB OK", new_tag, flush=True)
