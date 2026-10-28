"""Inference cost of the compressed models and the one-off cost of compressing them (one GPU to itself).

Usage:  python scripts/experiments/measure_cost.py <tag>
- Configurations: dense, and the paper's widths at 70% and 50% of feed-forward units. Speed depends only on the
  shapes, so each block keeps its first M_l units with the repaired consumer layout (down_proj with a bias row); every
  calibration condition of the paper has exactly these shapes.
- Prefill: one forward pass over batch x 512 tokens (batch 1 and 16). Decode: 128 new tokens after a 128-token
  prompt, greedy with the KV cache, batch 1 and 32 (the length is forced, end-of-text ignored). Median of 10 timed
  repeats after 3 warm-ups, CUDA-synchronized wall clock. Peak memory with torch.cuda.max_memory_allocated.
- Compression: wall clock for the statistics of one calibration set (128 x 512 tokens: 64 of C4, 64 of KG-D) at every
  block and the ridge repair of every block at 70% (the work for one condition of the paper).
Writes results/cost/<tag>_cost.json.
"""
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from kgcal import paths as P            # noqa: E402
from kgcal import compress as C        # noqa: E402
from kgcal import data as D            # noqa: E402
from kgcal import model_ops as MO      # noqa: E402

MODELS = {"qwen15b": "Qwen/Qwen2.5-1.5B", "olmo1b": "allenai/OLMo-2-0425-1B",
          "qwen7b": "Qwen/Qwen2.5-7B", "olmo7b": "allenai/OLMo-2-1124-7B"}
GROUP = {"qwen15b": 4, "olmo1b": 4, "qwen7b": 1, "olmo7b": 2}
tag = sys.argv[1]
OUT = P.COST
OUT.mkdir(parents=True, exist_ok=True)
log = lambda *a: print(time.strftime("%H:%M:%S"), f"[{tag}]", *a, flush=True)
model, tok = MO.load(MODELS[tag])
dev = next(model.parameters()).device
L, Nu, d = model.config.num_hidden_layers, model.config.intermediate_size, model.config.hidden_size
dense = MO.dense_copy(model)
dims = json.loads((P.GRID / f"{tag}_gride.json").read_text())["dims"]
WID = {f: MO.widths(dims, MO.alpha_for_fraction(dims, Nu, f), Nu) for f in (0.7, 0.5)}
vocab = model.config.vocab_size
g = torch.Generator(device="cpu").manual_seed(0)


def timed(fn, reps=10, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    return statistics.median(ts)


@torch.no_grad()
def measure(name):
    params = sum(p.numel() for p in model.parameters())
    ffn = sum(sum(getattr(m, k).weight.numel() for k in ("gate_proj", "up_proj", "down_proj")) for m in MO.mlps(model))
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    res = {"config": name, "params": params, "ffn_params": ffn, "weights_gb": torch.cuda.memory_allocated() / 2**30}
    for B in (1, 16):
        x = torch.randint(0, vocab, (B, 512), generator=g).to(dev)
        s = timed(lambda: model(input_ids=x, logits_to_keep=1))
        res[f"prefill_b{B}_s"], res[f"prefill_b{B}_tok_s"] = s, B * 512 / s
    for B in (1, 32):
        x = torch.randint(0, vocab, (B, 128), generator=g).to(dev)
        att = torch.ones_like(x)
        run = lambda: model.generate(input_ids=x, attention_mask=att, max_new_tokens=128, min_new_tokens=128,
                                     do_sample=False, pad_token_id=tok.pad_token_id)
        s = timed(run, reps=5, warm=2)
        res[f"decode_b{B}_s"], res[f"decode_b{B}_tok_s"] = s, B * 128 / s
        res[f"decode_b{B}_ms_per_token"] = 1000 * s / 128
    res["peak_gb"] = torch.cuda.max_memory_allocated() / 2**30
    log(name, {k: round(v, 3) if isinstance(v, float) else v for k, v in res.items()})
    return res


out = {"tag": tag, "model": MODELS[tag], "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
       "configs": [measure("dense")]}
for f in (0.7, 0.5):
    for i in range(L):
        m_ = WID[f][i]
        S = torch.arange(m_)
        A = C.consumer_none(dense[i]["down_proj"].float(), S).to(torch.bfloat16)      # W_S^T and a zero bias row
        MO.apply(model, dense, i, S, A, None)
    out["configs"].append(measure(f"{int(100 * f)}%") | {"ffn_keep": sum(WID[f]) / (L * Nu)})
    MO.restore(model, dense)

# one-off compression cost: statistics of one calibration set, then the ridge repair of every block at 70%
c4 = D.c4_lines(2500)
kg = [json.loads(l) for l in open(P.CALIB / "KG-D.jsonl")]
torch.cuda.synchronize(); t = time.perf_counter()
ids = torch.cat([D.pack(tok, c4, 128)[:64], D.pack(tok, kg, 128)[:64]])
t_pack = time.perf_counter() - t
t_stats = t_rep = 0.0
for g0 in range(0, L, GROUP[tag]):
    layers = list(range(g0, min(L, g0 + GROUP[tag])))
    torch.cuda.synchronize(); t = time.perf_counter()
    mom = MO.collect_moments(model, ids, layers)
    torch.cuda.synchronize(); t_stats += time.perf_counter() - t
    for i in layers:
        Wd = dense[i]["down_proj"].float()
        t = time.perf_counter()
        S = C.select("actnorm", Wd, WID[0.7][i], mom[i], dense[i])
        if len(S) < Nu:
            C.ridge(Wd, S, mom[i])
        torch.cuda.synchronize(); t_rep += time.perf_counter() - t
    del mom
out["compression"] = {"tokens": int(ids.numel()), "pack_s": t_pack, "statistics_s": t_stats, "select_repair_s": t_rep,
                      "total_min": (t_pack + t_stats + t_rep) / 60}
log("compression", out["compression"])
(OUT / f"{tag}_cost.json").write_text(json.dumps(out, indent=1))
print("JOB OK", tag, flush=True)
