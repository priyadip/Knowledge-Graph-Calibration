"""Transfer to a second pruning method: SparseGPT calibrated on C4 or on the paper's knowledge-graph mixtures.

Usage:  python scripts/experiments/sparsegpt.py <tag>
- SparseGPT as published (Frantar and Alistarh, 2023): every linear layer of every block (attention and feed-forward),
  50% of the weights of each layer (unstructured), OBS reconstruction with the Hessian of the layer's calibration
  inputs (dampening 1% of its mean diagonal, columns in blocks of 128). Blocks are pruned in order, each on the
  outputs of the already pruned blocks before it, and the linear layers of a block on the inputs of the unpruned block.
  Like the repair of the paper, the update depends on the calibration text only through second moments.
- Calibration: the paper's budget of 128 sequences of 512 tokens, seed 0. C4 is C4a + C4b (as for C4/C4) and MIX(X)
  is C4a + Xa (as for MIX(X)/MIX(X)), packed as in kgcal.run_grid from data/calib and data/calib_paired.
- Conditions: C4, MIX(KG-D), MIX(KG-D~), MIX(KG-QA), MIX(KG-QA~). Truth: KG-D vs KG-D~ and KG-QA vs KG-QA~ (paired).
- Evaluation exactly as run_grid at the generation batch of the original runs: validation and test questions, test
  cloze queries, WikiText-2 and held-out C4 perplexity. Retention uses the paper's dense run.
Writes results/transfer/<tag>_sgpt.jsonl and <tag>_sgpt_bits.json (resumes: finished conditions are skipped).
"""
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from kgcal import paths as P            # noqa: E402
from kgcal import data as D            # noqa: E402
from kgcal import evaluate as E        # noqa: E402
from kgcal import model_ops as MO      # noqa: E402

MODELS = {"qwen15b": "Qwen/Qwen2.5-1.5B", "olmo1b": "allenai/OLMo-2-0425-1B",
          "qwen7b": "Qwen/Qwen2.5-7B", "olmo7b": "allenai/OLMo-2-1124-7B"}
BATCH = {"qwen15b": 96, "olmo1b": 96, "qwen7b": 64, "olmo7b": 64}
CONDS = ["C4", "MIX:KG-D", "MIX:KG-D~", "MIX:KG-QA", "MIX:KG-QA~"]
SPARSITY = 0.5
tag = sys.argv[1]
OUT = P.SPARSEGPT
OUT.mkdir(parents=True, exist_ok=True)
res_path, bits_path = OUT / f"{tag}_sgpt.jsonl", OUT / f"{tag}_sgpt_bits.json"
t0 = time.time()
log = lambda *a: print(time.strftime("%H:%M:%S"), f"[{tag}]", *a, flush=True)


class Stop(Exception):
    pass


def sparsegpt_(lin: nn.Linear, H: torch.Tensor, sparsity: float, blocksize: int = 128, percdamp: float = 0.01):
    """In place: the SparseGPT reference algorithm (unstructured, adaptive mask per block of columns)."""
    Wt = lin.weight.data.clone().float()
    cols = Wt.shape[1]
    H = H.clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    Wt[:, dead] = 0
    H[range(cols), range(cols)] += percdamp * torch.mean(torch.diag(H))
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    for i1 in range(0, cols, blocksize):
        i2 = min(i1 + blocksize, cols)
        count = i2 - i1
        W1 = Wt[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
        thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
        mask1 = tmp <= thresh
        for i in range(count):
            w = W1[:, i]
            dd = Hinv1[i, i]
            q = w.clone()
            q[mask1[:, i]] = 0
            Q1[:, i] = q
            err1 = (w - q) / dd
            W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
            Err1[:, i] = err1
        Wt[:, i1:i2] = Q1
        Wt[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
    lin.weight.data = Wt.to(lin.weight.dtype)


@torch.no_grad()
def prune_model(model, ids: torch.Tensor, batch: int = 8):
    dev = next(model.parameters()).device
    layers = model.model.layers
    cache = {}

    def grab(mod, args, kwargs):
        cache.setdefault("h", []).append(args[0] if args else kwargs["hidden_states"])
        cache["kw"] = {k: v for k, v in kwargs.items() if k != "hidden_states"}
        raise Stop
    hk = layers[0].register_forward_pre_hook(grab, with_kwargs=True)
    for b in range(0, ids.shape[0], batch):
        try:
            model(input_ids=ids[b:b + batch].to(dev), use_cache=False)
        except Stop:
            pass
    hk.remove()
    hs, kw = cache["h"], cache["kw"]
    for li, layer in enumerate(layers):
        lins = {n: m for n, m in layer.named_modules() if isinstance(m, nn.Linear)}
        Hs = {n: torch.zeros(m.in_features, m.in_features, device=dev) for n, m in lins.items()}
        cnt = {n: 0 for n in lins}

        def mk(n):
            def h(mod, inp, out):
                x = inp[0].reshape(-1, inp[0].shape[-1]).float()
                k = inp[0].shape[0]
                Hs[n].mul_(cnt[n] / (cnt[n] + k))
                cnt[n] += k
                x = math.sqrt(2 / cnt[n]) * x
                Hs[n].addmm_(x.T, x)
            return h
        hooks = [m.register_forward_hook(mk(n)) for n, m in lins.items()]
        for h_ in hs:
            layer(h_, **kw)
        for h_ in hooks:
            h_.remove()
        for n, m in lins.items():
            sparsegpt_(m, Hs[n], SPARSITY)
        del Hs
        hs = [layer(h_, **kw) for h_ in hs]                      # inputs of the next block, after pruning
        torch.cuda.empty_cache()


model, tok = MO.load(MODELS[tag])
orig = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
rows = D.load_popqa()
F = D.facts_by_split(rows, D.make_split(rows))
ex = E.shots(F["cal"])
c4 = D.c4_lines(3000)
pp = E.ppl_sets(tok, c4[2600:3000])
c4_ids = D.pack(tok, c4[:2500], 128)
src = lambda k: [json.loads(l) for l in open(P.calib_file(k))]
CAL = {"C4": c4_ids}
for c in CONDS[1:]:
    CAL[c] = torch.cat([c4_ids[:64], D.pack(tok, src(c[4:]), 128)[:64]])
bits = json.loads(bits_path.read_text()) if bits_path.exists() else {}
for cond in CONDS:
    key = f"{SPARSITY}|sgpt|{cond}"
    if key in bits:
        continue
    t = time.time()
    model.load_state_dict(orig)
    prune_model(model, CAL[cond])
    t_prune = time.time() - t
    zero = sum(int((m.weight == 0).sum()) for l in model.model.layers for m in l.modules() if isinstance(m, nn.Linear))
    tot = sum(m.weight.numel() for l in model.model.layers for m in l.modules() if isinstance(m, nn.Linear))
    vq = E.recall(model, tok, F["val"], ex, "question", BATCH[tag])
    tq = E.recall(model, tok, F["test"], ex, "question", BATCH[tag])
    tc = E.recall(model, tok, F["test"], ex, "cloze", BATCH[tag])
    rec = {"tag": tag, "model": MODELS[tag], "method": "sparsegpt", "sparsity": SPARSITY, "calib": cond,
           "linear_zero_share": zero / tot, "val_q_acc": sum(vq) / len(vq), "test_q_acc": sum(tq) / len(tq),
           "test_c_acc": sum(tc) / len(tc), "ppl_wikitext2": E.perplexity(model, pp["wikitext2"]),
           "ppl_c4": E.perplexity(model, pp["c4"]), "prune_s": round(t_prune, 1), "secs": round(time.time() - t, 1),
           "batch": BATCH[tag]}
    bits[key] = {"val_q": "".join("1" if x else "0" for x in vq), "test_q": "".join("1" if x else "0" for x in tq),
                 "test_c": "".join("1" if x else "0" for x in tc)}
    with open(res_path, "a") as f:
        f.write(json.dumps(rec) + "\n")
    bits_path.write_text(json.dumps(bits))
    log(f"{key:28s} zero {rec['linear_zero_share']:.3f} test_q={rec['test_q_acc']:.4f} test_c={rec['test_c_acc']:.4f} "
        f"ppl_wt={rec['ppl_wikitext2']:.2f} (prune {t_prune:.0f}s, total {rec['secs']:.0f}s)")
log(f"all done in {time.time() - t0:.0f}s")
print("JOB OK", tag, flush=True)
