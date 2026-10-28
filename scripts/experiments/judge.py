"""A stronger judge for the saved answers: Phi-4 with every accepted answer and an explicit rule against echoes.

Usage:  python scripts/experiments/judge.py <tag>      (after score_answers.py)
Reads results/answers/<tag>_scores.json (written by score_answers.py) and adds the rule "judge": Phi-4 (14B,
microsoft/phi-4 at a pinned revision; the scored answers come from Qwen2.5 and OLMo-2) decides whether an answer names the same
entity as one of the accepted PopQA answers. Changes against the first judge (Llama-3.1-8B-Instruct, kept as
"judge_llama"), made after reading its disagreements with the alias rule: the judge sees every accepted answer (up to
600 characters) instead of the first 12, and the instruction says that an answer that only repeats a name from the
question is wrong (the first judge accepted "The Bottom" for "What is The Bottom the capital of?"). Decision: Yes
against No logit of the first answer token. The judge sees every dense answer and, for the compressed conditions,
the facts the dense model answers correctly under any rule.
"""
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from kgcal import paths as P            # noqa: E402
from kgcal import data as D            # noqa: E402

JUDGE = ("microsoft/phi-4", "2db69c1c3e91a05d2c64a3185acfbaf36f744e25")
tag = sys.argv[1]
t0 = time.time()
log = lambda *a: print(time.strftime("%H:%M:%S"), f"[{tag}]", *a, flush=True)
OUT = P.ANSWERS
rows = D.load_popqa()
TEST = D.facts_by_split(rows, D.make_split(rows))["test"]
GOLD = [D.ans(r) for r in TEST]
ANS = json.loads((P.PAIRED_CLOZE / f"{tag}_pc_answers.json").read_text())
SC = json.loads((OUT / f"{tag}_scores.json").read_text())
R = SC["rules"]
if "judge" in R and "judge_llama" not in R:
    R["judge_llama"] = R.pop("judge")
    SC["judge_llama_p_yes"] = SC.pop("judge_p_yes")
CONDS = SC["conds"]
first = lambda t: t.strip().split("\n")[0]

jt = AutoTokenizer.from_pretrained(JUDGE[0], revision=JUDGE[1])
jm = AutoModelForCausalLM.from_pretrained(JUDGE[0], revision=JUDGE[1], dtype=torch.bfloat16).cuda().eval()
jt.padding_side = "left"
YES, NO = jt.encode("Yes", add_special_tokens=False), jt.encode("No", add_special_tokens=False)
assert len(YES) == 1 and len(NO) == 1
YES, NO = YES[0], NO[0]


def accepted(g):
    out, n = [], 0
    for a in g:
        if n + len(a) > 600:
            break
        out.append(a); n += len(a) + 2
    return "; ".join(out)


def prompt(i, answer, fmt):
    q = TEST[i]["question"] if fmt == "test_q" else D.cloze(TEST[i]["prop"], TEST[i]["subj"]) + " ..."
    msg = (f"Question: {q}\nAccepted answers (any one of them is correct): {accepted(GOLD[i])}\n"
           f"Model answer: {answer.strip() or '(empty)'}\n\n"
           "Is the model answer correct? It is correct if it names the same entity as one of the accepted answers, even "
           "with different spelling, wording, a more specific or a less specific name of the same entity, or extra "
           "detail. It is wrong if it names a different entity, gives no answer, or only repeats a name that appears in "
           "the question. Reply with Yes or No only.")
    return jt.apply_chat_template([{"role": "system", "content": "You grade short answers to factual questions."},
                                   {"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def judge(items, batch=16):
    """P(Yes) for every item; prompts are batched in order of length (less padding), results in input order."""
    ps = [prompt(i, a, f) for i, a, f in items]
    lens = [len(jt(p, add_special_tokens=False)["input_ids"]) for p in ps]
    order = sorted(range(len(ps)), key=lambda k: lens[k])
    out = [None] * len(ps)
    for b in range(0, len(order), batch):
        idx = order[b:b + batch]
        enc = jt([ps[k] for k in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        lg = jm(**enc, logits_to_keep=1).logits[:, -1, :].float()
        for k, v in zip(idx, torch.softmax(lg[:, [YES, NO]], -1)[:, 0].tolist()):
            out[k] = v
    return out


R["judge"] = {c: {} for c in CONDS}
PY = {c: {} for c in CONDS}
for f in ("test_q", "test_c"):
    p = judge([(i, first(ANS["1.0|dense"][f][i]), f) for i in range(len(TEST))])
    PY["1.0|dense"][f] = p
    R["judge"]["1.0|dense"][f] = "".join("1" if x >= 0.5 else "0" for x in p)
    rules = [r for r in R if r != "judge"]
    known_any = [i for i in range(len(TEST)) if p[i] >= 0.5 or any(R[r]["1.0|dense"][f][i] == "1" for r in rules)]
    for c in CONDS[1:]:
        q = judge([(i, first(ANS[c][f][i]), f) for i in known_any])
        full, prob = ["-"] * len(TEST), [None] * len(TEST)
        for i, x in zip(known_any, q):
            full[i], prob[i] = ("1" if x >= 0.5 else "0"), x
        R["judge"][c][f], PY[c][f] = "".join(full), prob
    log(f"Phi-4 judge {f}: {len(TEST)} dense answers, {len(known_any)} facts per condition ({time.time() - t0:.0f}s)")
SC["judge_p_yes"], SC["judge"] = PY, JUDGE
SC["judge_llama"] = ("unsloth/Meta-Llama-3.1-8B-Instruct", "a2856192dd7c25b842431f39c179a6c2c2f627d1")
agree = {f: sum(a == b for a, b in zip(R["alias"]["1.0|dense"][f], R["judge"]["1.0|dense"][f])) / len(TEST)
         for f in ("test_q", "test_c")}
log("Phi-4 judge agreement with the alias rule on dense answers:", agree)
(OUT / f"{tag}_scores.json").write_text(json.dumps(SC))
log(f"all done in {time.time() - t0:.0f}s")
print("JOB OK", tag, flush=True)
