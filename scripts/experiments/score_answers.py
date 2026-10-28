"""Score the saved answers of the paired cloze runs with four rules, and the judge's agreement with the paper's rule.

Usage:  python scripts/experiments/score_answers.py <tag> [--no-judge]
Answers: results/paired_cloze/<tag>_pc_answers.json (the decoded continuation of every test question and
cloze query, for the dense model and the eight conditions at 70% and 50%).
Rules, each applied to the first generated line:
- alias: the paper's rule, any PopQA alias occurs in it (case-insensitive). Must reproduce the saved bits exactly.
- word: an alias occurs as whole words (no match inside a longer word, such as "art" in "Arthur").
- exact: the answer, cut at the first comma, period, semicolon or parenthesis and normalized (case, articles,
  punctuation, spaces), equals a normalized alias.
- judge: Llama-3.1-8B-Instruct (the scored answers come from Qwen2.5 and OLMo-2) decides whether the answer gives a correct
  answer given the question and the list of correct answers (one forward pass, Yes against No logit). The judge sees
  every test answer of the dense model and, for the compressed conditions, the facts the dense model answers
  correctly under any rule (retention needs no other facts).
Writes results/answers/<tag>_scores.json: per rule, condition and query format, a 0/1 string over the test facts
("-" where a fact was not judged), plus the judge's Yes probability.
"""
import json
import re
import string
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from kgcal import paths as P            # noqa: E402
from kgcal import data as D            # noqa: E402

JUDGE = ("unsloth/Meta-Llama-3.1-8B-Instruct", "a2856192dd7c25b842431f39c179a6c2c2f627d1")
tag = sys.argv[1]
use_judge = "--no-judge" not in sys.argv
t0 = time.time()
log = lambda *a: print(time.strftime("%H:%M:%S"), f"[{tag}]", *a, flush=True)
PC = P.PAIRED_CLOZE
OUT = P.ANSWERS
OUT.mkdir(parents=True, exist_ok=True)

rows = D.load_popqa()
TEST = D.facts_by_split(rows, D.make_split(rows))["test"]
GOLD = [D.ans(r) for r in TEST]
ANS = json.loads((PC / f"{tag}_pc_answers.json").read_text())
BITS = json.loads((PC / f"{tag}_pc_bits.json").read_text())
CONDS = list(ANS)
assert CONDS[0] == "1.0|dense" and all(len(ANS[c]["test_q"]) == len(TEST) for c in CONDS)

first = lambda t: t.strip().split("\n")[0]
ARTICLES = re.compile(r"\b(a|an|the)\b")
PUNCT = str.maketrans("", "", string.punctuation)


def norm(s: str) -> str:
    s = ARTICLES.sub(" ", s.lower()).translate(PUNCT)
    return " ".join(s.split())


def rule_alias(t, gold):
    line = first(t).lower()
    return any(a.lower() in line for a in gold)


def rule_word(t, gold):
    line = first(t).lower()
    return any(re.search(r"(?<!\w)" + re.escape(a.lower()) + r"(?!\w)", line) for a in gold)


def rule_exact(t, gold):
    span = re.split(r"[,.;(]", first(t), maxsplit=1)[0]
    return norm(span) in {norm(a) for a in gold}


RULES = {"alias": rule_alias, "word": rule_word, "exact": rule_exact}
S = {r: {c: {} for c in CONDS} for r in list(RULES) + (["judge"] if use_judge else [])}
for c in CONDS:
    for f in ("test_q", "test_c"):
        for rn, fn in RULES.items():
            S[rn][c][f] = "".join("1" if fn(t, g) else "0" for t, g in zip(ANS[c][f], GOLD))
        assert S["alias"][c][f] == BITS[c][f], f"alias rule does not reproduce the saved bits for {c} {f}"
log("rules alias/word/exact done; alias reproduces the saved bits for all", len(CONDS), "conditions")

if use_judge:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    jt = AutoTokenizer.from_pretrained(JUDGE[0], revision=JUDGE[1])
    jm = AutoModelForCausalLM.from_pretrained(JUDGE[0], revision=JUDGE[1], dtype=torch.bfloat16).cuda().eval()
    jt.padding_side = "left"
    if jt.pad_token is None:
        jt.pad_token = jt.eos_token
    yes = jt.encode("Yes", add_special_tokens=False); no = jt.encode("No", add_special_tokens=False)
    assert len(yes) == 1 and len(no) == 1
    YES, NO = yes[0], no[0]

    def prompt(i, answer, fmt):
        q = TEST[i]["question"] if fmt == "test_q" else D.cloze(TEST[i]["prop"], TEST[i]["subj"]) + " ..."
        msg = (f"Question: {q}\nCorrect answers (any one of these is correct): {'; '.join(GOLD[i][:12])}\n"
               f"Answer to grade: {answer.strip() or '(empty)'}\n"
               "Does the answer to grade give a correct answer to the question? Small differences in spelling, "
               "wording or detail are fine, and extra correct detail is fine. Reply with Yes or No only.")
        return jt.apply_chat_template([{"role": "system", "content": "You grade short answers to factual questions."},
                                       {"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True)

    @torch.no_grad()
    def judge(items, batch=96):
        """items: (fact index, answer line, format) -> P(Yes) among {Yes, No}."""
        out = []
        for b in range(0, len(items), batch):
            ps = [prompt(i, a, f) for i, a, f in items[b:b + batch]]
            enc = jt(ps, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            lg = jm(**enc, logits_to_keep=1).logits[:, -1, :].float()
            out.extend(torch.softmax(lg[:, [YES, NO]], -1)[:, 0].tolist())
        return out

    PY = {c: {} for c in CONDS}
    for f in ("test_q", "test_c"):
        p = judge([(i, first(ANS["1.0|dense"][f][i]), f) for i in range(len(TEST))])
        PY["1.0|dense"][f] = p
        known_any = [i for i in range(len(TEST)) if p[i] >= 0.5 or any(S[r]["1.0|dense"][f][i] == "1" for r in RULES)]
        S["judge"]["1.0|dense"][f] = "".join("1" if x >= 0.5 else "0" for x in p)
        for c in CONDS[1:]:
            q = judge([(i, first(ANS[c][f][i]), f) for i in known_any])
            full, prob = ["-"] * len(TEST), [None] * len(TEST)
            for i, x in zip(known_any, q):
                full[i], prob[i] = ("1" if x >= 0.5 else "0"), x
            S["judge"][c][f], PY[c][f] = "".join(full), prob
        log(f"judge {f}: {len(TEST)} dense answers, {len(known_any)} facts for each of {len(CONDS) - 1} conditions "
            f"({time.time() - t0:.0f}s)")
    agree = {f: float(np.mean([a == b for a, b in zip(S["alias"]["1.0|dense"][f], S["judge"]["1.0|dense"][f])]))
             for f in ("test_q", "test_c")}
    log("judge agreement with the alias rule on dense answers:", agree)
else:
    PY = None
(OUT / f"{tag}_scores.json").write_text(json.dumps({"tag": tag, "conds": CONDS, "rules": S, "judge_p_yes": PY,
                                                    "judge": JUDGE}))
log(f"all done in {time.time() - t0:.0f}s")
print("JOB OK", tag, flush=True)
