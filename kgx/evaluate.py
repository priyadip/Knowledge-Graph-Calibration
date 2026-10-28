"""kgcal.evaluate with the start-of-text token of BOS families.

Question and cloze recall, perplexity and the zero-shot item sets are kgcal.evaluate unchanged: generation prompts
are tokenized with the tokenizer's defaults, so a BOS family gets BOS there already. Two token sequences are built
by hand in kgcal and get BOS here when the family uses one: the perplexity windows (BOS + 1,023 text tokens; in kgcal
only the first window of the stream would start with BOS) and the zero-shot contexts. With bos = None both functions
call kgcal unchanged.
"""
from __future__ import annotations

import torch

from kgcal import evaluate as KE
from kgcal.data import HF_REVISIONS as REV
from kgcal.evaluate import perplexity, prompts, recall, shots, zeroshot_sets  # noqa: F401  (unchanged)


def ppl_sets(tok, c4_heldout: list[str], n_seq: int = 64, seq_len: int = 1024, bos: int | None = None):
    if bos is None:
        return KE.ppl_sets(tok, c4_heldout, n_seq, seq_len)
    from datasets import load_dataset
    wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test", revision=REV["Salesforce/wikitext"])
    body = seq_len - 1

    def windows(text):
        ids = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0][: n_seq * body].view(n_seq, body)
        return torch.cat([torch.full((n_seq, 1), bos, dtype=ids.dtype), ids], 1)
    return {"wikitext2": windows("\n\n".join(wt["text"])), "c4": windows("\n".join(c4_heldout))}


@torch.no_grad()
def multiple_choice(model, tok, items, batch: int = 32, length_norm: bool = True, bos: int | None = None) -> float:
    """kgcal.evaluate.multiple_choice; with a BOS family the context is BOS + context tokens."""
    if bos is None:
        return KE.multiple_choice(model, tok, items, batch, length_norm)
    dev = next(model.parameters()).device
    tok.padding_side = "right"
    pairs = [(ctx, ch, gi, k) for gi, (ctx, chs, _) in enumerate(items) for k, ch in enumerate(chs)]
    scores = {}
    for b in range(0, len(pairs), batch):
        chunk = pairs[b:b + batch]
        ctx_ids = [[bos] + tok(c, add_special_tokens=False)["input_ids"] for c, _, _, _ in chunk]
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
