"""Fact recall (question and cloze queries with alias matching) and generic quality (perplexity, zero-shot)."""
from __future__ import annotations

import math
import random

import torch

from .data import HF_REVISIONS as REV
from .data import ans, cloze, declarative


def shots(cal_facts: list[dict], k: int = 4, seed: int = 0) -> list[dict]:
    """k in-context examples from calibration entities, each from a different relation."""
    rng = random.Random(seed)
    pool = cal_facts[:]
    rng.shuffle(pool)
    out, seen = [], set()
    for r in pool:
        if r["prop"] not in seen:
            out.append(r)
            seen.add(r["prop"])
        if len(out) == k:
            break
    return out


def prompts(facts: list[dict], examples: list[dict], fmt: str) -> list[str]:
    if fmt == "question":
        pre = "".join(f"Q: {e['question']} A: {ans(e)[0]}\n" for e in examples)
        return [pre + f"Q: {r['question']} A:" for r in facts]
    if fmt == "cloze":
        pre = "".join(declarative(e["prop"], e["subj"], ans(e)[0]) + "\n" for e in examples)
        return [pre + cloze(r["prop"], r["subj"]) for r in facts]
    raise ValueError(fmt)


@torch.no_grad()
def recall(model, tok, facts: list[dict], examples: list[dict], fmt: str, batch: int = 96,
           max_new: int = 12) -> list[bool]:
    """Greedy generation; correct if any PopQA answer alias occurs (case-insensitive) in the first output line."""
    dev = next(model.parameters()).device
    tok.padding_side = "left"
    ps = prompts(facts, examples, fmt)
    out: list[bool] = []
    for b in range(0, len(ps), batch):
        enc = tok(ps[b:b + batch], return_tensors="pt", padding=True).to(dev)
        g = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        texts = tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for r, t in zip(facts[b:b + batch], texts):
            line = t.strip().split("\n")[0].lower()
            out.append(any(a.lower() in line for a in ans(r)))
    return out


@torch.no_grad()
def perplexity(model, ids: torch.Tensor, batch: int = 2) -> float:
    dev = next(model.parameters()).device
    nll, n = 0.0, 0
    for b in range(0, ids.shape[0], batch):
        x = ids[b:b + batch].to(dev)
        loss = model(input_ids=x, labels=x).loss
        nll += loss.item() * x.shape[0]
        n += x.shape[0]
    return math.exp(nll / n)


def ppl_sets(tok, c4_heldout: list[str], n_seq: int = 64, seq_len: int = 1024) -> dict[str, torch.Tensor]:
    from datasets import load_dataset
    wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test", revision=REV["Salesforce/wikitext"])
    wt_ids = tok("\n\n".join(wt["text"]), return_tensors="pt")["input_ids"][0][: n_seq * seq_len].view(n_seq, seq_len)
    c4_ids = tok("\n".join(c4_heldout), return_tensors="pt")["input_ids"][0][: n_seq * seq_len].view(n_seq, seq_len)
    return {"wikitext2": wt_ids, "c4": c4_ids}


# --------------------------------------------------------------------------- zero-shot multiple choice
def zeroshot_sets(n_hella: int = 2000, seed: int = 0) -> dict[str, list[tuple[str, list[str], int]]]:
    """ARC-Easy (test) and a fixed HellaSwag validation subset as (context, choices, gold index)."""
    from datasets import load_dataset
    arc = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test", revision=REV["allenai/ai2_arc"])
    arc_items = []
    for r in arc:
        labels = r["choices"]["label"]
        if r["answerKey"] not in labels:
            continue
        arc_items.append((f"Question: {r['question']}\nAnswer:", [" " + t for t in r["choices"]["text"]],
                          labels.index(r["answerKey"])))
    hs = load_dataset("Rowan/hellaswag", split="validation", revision=REV["Rowan/hellaswag"])
    idx = random.Random(seed).sample(range(len(hs)), n_hella)
    hs_items = []
    for i in idx:
        r = hs[i]
        ctx = (r["activity_label"] + ": " + r["ctx_a"] + " " + r["ctx_b"].capitalize()).strip()
        hs_items.append((ctx, [" " + e for e in r["endings"]], int(r["label"])))
    return {"arc_easy": arc_items, "hellaswag": hs_items}


@torch.no_grad()
def multiple_choice(model, tok, items, batch: int = 32, length_norm: bool = True) -> float:
    """Accuracy by (length-normalized) log-likelihood of each choice given the context."""
    dev = next(model.parameters()).device
    tok.padding_side = "right"
    pairs = [(ctx, ch, gi, k) for gi, (ctx, chs, _) in enumerate(items) for k, ch in enumerate(chs)]
    scores = {}
    for b in range(0, len(pairs), batch):
        chunk = pairs[b:b + batch]
        ctx_ids = [tok(c, add_special_tokens=False)["input_ids"] for c, _, _, _ in chunk]
        ch_ids = [tok(ch, add_special_tokens=False)["input_ids"] for _, ch, _, _ in chunk]
        seqs = [a + c for a, c in zip(ctx_ids, ch_ids)]
        T = max(len(s) for s in seqs)
        x = torch.full((len(seqs), T), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(seqs), T), dtype=torch.long)
        for i, s in enumerate(seqs):
            x[i, :len(s)] = torch.tensor(s)
            att[i, :len(s)] = 1
        logp = torch.log_softmax(model(input_ids=x.to(dev), attention_mask=att.to(dev)).logits.float(), -1)
        for i, ((_, _, gi, k), a, c) in enumerate(zip(chunk, ctx_ids, ch_ids)):
            pos = torch.arange(len(a) - 1, len(a) + len(c) - 1, device=dev)
            tgt = torch.tensor(c, device=dev)
            lp = logp[i, pos, tgt].sum().item()
            scores[(gi, k)] = lp / len(c) if length_norm else lp
    correct = 0
    for gi, (_, chs, gold) in enumerate(items):
        pred = max(range(len(chs)), key=lambda k: scores[(gi, k)])
        correct += int(pred == gold)
    return correct / len(items)
