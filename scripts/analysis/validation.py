"""Two checks of how far the inference and the theory reach (read-only on existing results).

Usage:  python scripts/analysis/validation.py      (before scripts/paper/make_numbers.py)
1. Subject-clustered intervals. Several test facts share a subject, so fact-level resampling may understate the
   uncertainty. For every setting we recompute the paired contrasts with a bootstrap that resamples test subjects with
   all their facts (10,000 draws, seed 0, one draw of subjects shared by all settings since every run scores the same
   facts) and compare with the fact-level bootstrap of the paper. The pooled truth effect is recomputed with the same
   two-level scheme as stats.py (calibration samples resampled within each setting), with subjects in place of
   facts at the first level.
2. Local repair error and retention. Lemma 1 is exact per block; its link to end-to-end retention is empirical.
   For every pair of consumers that share the kept units (same model, budget and selection text), we ask whether the
   consumer with the lower error ratio on knowledge-question contexts (mean over measured blocks, from the theory
   files) also keeps more test facts. Ties in either quantity are skipped. A two-sided sign test compares the share of
   concordant pairs with one half. Qwen2.5-7B has no Lemma 1 measurements.
Writes outputs/tables/validation.json.
"""
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from kgcal import paths as P            # noqa: E402
from kgcal import data as D                     # noqa: E402

MODELS = ["qwen15b", "olmo1b", "qwen7b", "olmo7b", "llama8b", "mistral7b", "phi4"]
PAPER = MODELS[:4]
BUD = ("0.7", "0.5")
NB = 10000
src = lambda t: P.MERGED if t in PAPER else P.FAMILIES
K = {"c4": "C4|C4", "rep": "C4|MIX:KG-D", "sel": "MIX:KG-D|C4", "both": "MIX:KG-D|MIX:KG-D",
     "dcf": "MIX:KG-D~|MIX:KG-D~", "qa": "MIX:KG-QA|MIX:KG-QA", "qacf": "MIX:KG-QA~|MIX:KG-QA~",
     "c4none": "none|C4", "selnone": "MIX:KG-D|none"}           # the result files key C4/none as "none|C4"
CONTRASTS = {"C4 repair vs none": ("c4none", "c4"), "Repair channel": ("c4", "rep"), "Selection channel": ("c4", "sel"),
             "Selection, no repair": ("c4none", "selnone"), "Both channels": ("c4", "both"),
             "Truth (declarative)": ("dcf", "both"), "Truth (question-answer)": ("qacf", "qa")}
OUT = {}

rows = D.load_popqa()
test = D.facts_by_split(rows, D.make_split(rows))["test"]
_, sid = np.unique([D.qid(r["s_uri"]) for r in test], return_inverse=True)
n_subj = int(sid.max() + 1)
rng = np.random.default_rng(0)
w_subj = np.stack([np.bincount(rng.integers(0, n_subj, n_subj), minlength=n_subj) for _ in range(NB)]).astype(np.float32)
fact_draws = np.random.default_rng(0)


def bits(tag, s=0):
    name = f"{tag}_bits.json" if s == 0 else f"{tag}_cs{s}_bits.json"
    return json.loads((src(tag) / name).read_text())


vec = lambda b, k: np.array([c == "1" for c in b[k]["test_q"]], np.float32)


def clustered(d, m):
    """Subject-level replicates of the mean of d over the facts in m (points)."""
    num = np.bincount(sid, weights=d * m, minlength=n_subj).astype(np.float32)
    den = np.bincount(sid, weights=m, minlength=n_subj).astype(np.float32)
    return 100 * (w_subj @ num) / (w_subj @ den)


def fact_level(d, m):
    x = d[m.astype(bool)]
    idx = fact_draws.integers(0, len(x), size=(NB, len(x)))
    return 100 * x[idx].mean(1)


# ---------------------------------------------------------------- 1. subject-clustered intervals
cells, sample0_reps = {}, {}
for tag in MODELS:
    b = bits(tag)
    m = vec(b, "1.0|dense")
    for t in BUD:
        for name, (x, y) in CONTRASTS.items():
            d = vec(b, f"{t}|{K[y]}") - vec(b, f"{t}|{K[x]}")
            rep_c = clustered(d, m)
            sample0_reps[(tag, t, name)] = rep_c
            fl, cl = np.quantile(fact_level(d, m), [0.025, 0.975]), np.quantile(rep_c, [0.025, 0.975])
            cells[f"{tag}|{t}|{name}"] = {"diff": float(100 * (d * m).sum() / m.sum()),
                                          "fact_lo": float(fl[0]), "fact_hi": float(fl[1]),
                                          "subj_lo": float(cl[0]), "subj_hi": float(cl[1])}
summary = {}
for name in CONTRASTS:
    cs = [cells[f"{tag}|{t}|{name}"] for tag in MODELS for t in BUD]
    sig = lambda c, k: c[k + "_lo"] > 0 or c[k + "_hi"] < 0
    summary[name] = {"settings": len(cs), "sig_fact": sum(sig(c, "fact") for c in cs),
                     "sig_subj": sum(sig(c, "subj") for c in cs),
                     "sig_subj_pos": sum(c["subj_lo"] > 0 for c in cs),
                     "width_ratio": float(np.mean([(c["subj_hi"] - c["subj_lo"]) / (c["fact_hi"] - c["fact_lo"])
                                                   for c in cs]))}
OUT["clustered"] = {"subjects": n_subj, "facts": len(test), "cells": cells, "summary": summary}

# pooled truth effect, subjects resampled at the first level and calibration samples within each setting
pooled = {}
groups = {"paper": PAPER, "new": MODELS[4:], "all": MODELS}
reps_by_cell, est_by_cell = {}, {}
for tag in MODELS:
    bs = [bits(tag, s) for s in (0, 1, 2)]
    m = vec(bs[0], "1.0|dense")
    for t in BUD:
        for name in ("Truth (declarative)", "Truth (question-answer)"):
            x, y = CONTRASTS[name]
            ds = [vec(b, f"{t}|{K[y]}") - vec(b, f"{t}|{K[x]}") for b in bs]
            est_by_cell[(tag, t, name)] = float(np.mean([100 * (d * m).sum() / m.sum() for d in ds]))
            reps_by_cell[(tag, t, name)] = np.stack([clustered(d, m) for d in ds], 1)
pick_rng = np.random.default_rng(1)
for g, tags in groups.items():
    keys = [k for k in reps_by_cell if k[0] in tags]
    pick = pick_rng.integers(0, 3, size=(NB, len(keys), 3))
    rep = np.mean([np.take_along_axis(reps_by_cell[k], pick[:, i, :], 1).mean(1) for i, k in enumerate(keys)], 0)
    lo, hi = np.quantile(rep, [0.025, 0.975])
    lo90, hi90 = np.quantile(rep, [0.05, 0.95])
    pooled[g] = {"mean": float(np.mean([est_by_cell[k] for k in keys])), "lo": float(lo), "hi": float(hi),
                 "lo90": float(lo90), "hi90": float(hi90), "cells": len(keys)}
OUT["clustered"]["pooled_truth"] = pooled
pooled_contrast = {}
for name in CONTRASTS:
    rep = np.mean([sample0_reps[(tag, t, name)] for tag in MODELS for t in BUD], 0)
    lo, hi = np.quantile(rep, [0.025, 0.975])
    pooled_contrast[name] = {"mean": float(np.mean([cells[f"{tag}|{t}|{name}"]["diff"] for tag in MODELS for t in BUD])),
                             "lo": float(lo), "hi": float(hi)}
for name in ("Truth (declarative)", "Truth (question-answer)"):   # three calibration samples, as in the text
    keys = [k for k in reps_by_cell if k[2] == name]
    pick = np.random.default_rng(2).integers(0, 3, size=(NB, len(keys), 3))
    rep = np.mean([np.take_along_axis(reps_by_cell[k], pick[:, i, :], 1).mean(1) for i, k in enumerate(keys)], 0)
    lo, hi = np.quantile(rep, [0.025, 0.975])
    pooled_contrast[name + ", three samples"] = {"mean": float(np.mean([est_by_cell[k] for k in keys])),
                                                 "lo": float(lo), "hi": float(hi)}
OUT["clustered"]["pooled_contrast"] = pooled_contrast
pooled_group = {}
for g, tags in groups.items():
    for name in ("Both channels", "Repair channel"):
        rep = np.mean([sample0_reps[(tag, t, name)] for tag in tags for t in BUD], 0)
        lo, hi = np.quantile(rep, [0.025, 0.975])
        pooled_group[f"{g}|{name}"] = {"mean": float(np.mean([cells[f"{tag}|{t}|{name}"]["diff"] for tag in tags for t in BUD])),
                                       "lo": float(lo), "hi": float(hi)}
OUT["clustered"]["pooled_group"] = pooled_group
truth_setting = {}
pick = np.random.default_rng(3).integers(0, 3, size=(NB, 2, 3))
for tag in MODELS:
    for t in BUD:
        ks = [(tag, t, "Truth (declarative)"), (tag, t, "Truth (question-answer)")]
        rep = np.mean([np.take_along_axis(reps_by_cell[k], pick[:, i, :], 1).mean(1) for i, k in enumerate(ks)], 0)
        lo, hi = np.quantile(rep, [0.025, 0.975])
        truth_setting[f"{tag}|{t}"] = {"diff": float(np.mean([est_by_cell[k] for k in ks])), "lo": float(lo), "hi": float(hi)}
OUT["clustered"]["truth_setting_both_formats"] = truth_setting
truth_cell = {}                                                   # per setting and format, three samples
pick = np.random.default_rng(4).integers(0, 3, size=(NB, 3))
for k, r in reps_by_cell.items():
    lo, hi = np.quantile(np.take_along_axis(r, pick, 1).mean(1), [0.025, 0.975])
    truth_cell["|".join(k)] = {"diff": est_by_cell[k], "lo": float(lo), "hi": float(hi)}
OUT["clustered"]["truth_cell_three_samples"] = truth_cell

# ---------------------------------------------------------------- 2. local repair error and retention
THEORY = ["qwen15b", "olmo1b", "olmo7b", "llama8b", "mistral7b", "phi4"]


def cond(sel, cons):
    s = sel.split(":", 1)[1]
    if cons == "none":                        # the result files key C4 selection without repair as "none|C4"
        return "none|C4" if s == "C4" else f"{s}|none"
    return f"{s}|{cons.split(':', 1)[1]}"


pairs, per_model, per_setting, examples = [], {}, {}, []
for tag in THEORY:
    th = [json.loads(l) for l in open(src(tag) / f"{tag}_theory.jsonl")]
    b = bits(tag)
    m = vec(b, "1.0|dense").astype(bool)
    ret = lambda k: float(100 * vec(b, k)[m].mean())
    for t in (0.7, 0.5):
        rs_t = [r for r in th if r["target"] == t and r.get("question")]
        for sel in sorted({r["sel"] for r in rs_t if r["sel"].startswith("actnorm:")}):
            rs = [r for r in rs_t if r["sel"] == sel]
            cons = sorted({c for r in rs for c in r["question"] if c != "unavoidable"})
            pts = []
            for c in cons:
                k = f"{t}|{cond(sel, c)}"
                if k in b and all(c in r["question"] for r in rs):
                    pts.append((float(np.mean([r["question"][c]["ratio"] for r in rs])), ret(k), cond(sel, c)))
            for (ea, ra, na), (eb, rb, nb) in itertools.combinations(pts, 2):
                if abs(ea - eb) < 1e-9 or abs(ra - rb) < 1e-9:
                    continue
                ok = (ea < eb) == (ra > rb)
                pairs.append(ok)
                per_model.setdefault(tag, [0, 0])[0 if ok else 1] += 1
                per_setting.setdefault(f"{tag}|{t}", [0, 0])[0 if ok else 1] += 1
                if not ok:
                    examples.append({"model": tag, "budget": t, "a": na, "b": nb, "err": [ea, eb], "ret": [ra, rb]})
n, k = len(pairs), int(sum(pairs))
p = min(1.0, 2 * sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n)
OUT["local_error"] = {"pairs": n, "concordant": k, "share": k / n, "sign_test_p": p,
                      "per_model": {t: {"concordant": v[0], "pairs": v[0] + v[1]} for t, v in per_model.items()},
                      "per_setting": {k: {"concordant": v[0], "pairs": v[0] + v[1]} for k, v in per_setting.items()},
                      "discordant": examples}

(P.TABLES / "validation.json").write_text(json.dumps(OUT, indent=1))
for name, s in summary.items():
    print(f"{name}: fact-level sig {s['sig_fact']}/14, subject-clustered sig {s['sig_subj']}/14 "
          f"(positive {s['sig_subj_pos']}), width ratio {s['width_ratio']:.3f}")
for name, v in pooled_contrast.items():
    print(f"pooled {name}: {v['mean']:+.2f} [{v['lo']:+.2f}, {v['hi']:+.2f}]")
for g, v in pooled.items():
    print(f"pooled truth {g}: {v['mean']:+.2f} [{v['lo']:+.2f}, {v['hi']:+.2f}], 90% [{v['lo90']:+.2f}, {v['hi90']:+.2f}]")
for name in ("Truth (declarative)", "Truth (question-answer)"):
    cs = [v for k, v in truth_cell.items() if k.endswith(name)]
    print(f"{name}, three samples: range {min(c['diff'] for c in cs):+.1f} to {max(c['diff'] for c in cs):+.1f}, "
          f"clustered sig + {sum(c['lo'] > 0 for c in cs)}, - {sum(c['hi'] < 0 for c in cs)}")
print(f"local error vs retention: {k}/{n} concordant ({100 * k / n:.0f}%), sign test p = {p:.1e}")
print({t: f"{v[0]}/{v[0] + v[1]}" for t, v in per_model.items()})
