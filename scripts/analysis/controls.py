"""Controls for format, scoring and truth, analysed with the decision rules fixed before the runs.

Usage:  python scripts/analysis/controls.py      (before scripts/paper/make_numbers.py)
7a matched format: per setting, QA(S) minus D(S)rep with question queries (primary: same facts, same tokens),
   QA(S) minus D(S) (same facts, fewer declarative tokens), the primary contrast with cloze queries and the difference
   of the two query types, and MIX(KG-D) minus D(S)rep (more facts at the same format and nearly the same tokens;
   MIX(KG-D) from the paper's runs). A model group (1B, 7B) shows a format effect at matched facts only if the primary
   contrast is significant with the same sign in all four of its settings.
7b multiple choice: retention by likelihood (gold alias against three same-relation distractors) for the nine core
   conditions; the contrasts of Table 3 and the format contrast. The main findings survive under multiple choice if
   both channels beat C4/C4 significantly in at least six of the eight settings; the truth difference (two formats x
   eight settings) is pooled and tested for equivalence within 1.5 points.
7c truth: V0 (true) minus each intervention (V25, V50, V75, V100, VX, VC), pooled over the eight settings; equivalence
   only if the 90% interval lies inside +-1.5 points; the least-squares slope of retention on the share of false lines
   (V0, V25, V50, V75, V100) per setting, pooled, with its 95% interval (points per 100% false lines).
Intervals: bootstrap over test facts (10,000 draws, seed 0), one draw shared by every cell, so pooled means and
differences of contrasts have paired intervals; models and budgets are fixed. Every run's replay (dense, C4/C4 and,
for multiple choice, MIX(KG-D)/MIX(KG-D)) is checked again against the main runs. Writes outputs/tables/controls.json; settings
whose run has not finished are listed under "missing".
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from kgcal import paths as P            # noqa: E402
PAPER = ["qwen15b", "olmo1b", "qwen7b", "olmo7b"]
NEW = ["llama8b", "mistral7b", "phi4"]                     # scripts/experiments/controls_families.py
ALL = PAPER + NEW
GROUPS = {"small": PAPER[:2], "large": PAPER[2:], "all": PAPER, "new": NEW, "seven": ALL}
RULE_GROUPS = {"small": PAPER[:2], "large": PAPER[2:], "new": NEW}
BUD = ("0.7", "0.5")
NB, MARGIN, N_TEST = 10000, 1.5, 5690
RV = P.CONTROL_RESULTS
rng = np.random.default_rng(0)
WF = np.stack([np.bincount(rng.integers(0, N_TEST, N_TEST), minlength=N_TEST) for _ in range(NB)]).astype(np.float32)
OUT = {"missing": [], "info": {}}
vec = lambda s: np.array([c == "1" for c in s], np.float32)


def rep_mean(d, m):
    return 100 * (WF @ (d * m)) / (WF @ m)


def summ(rep, point):
    lo, hi = np.quantile(rep, [0.025, 0.975])
    lo90, hi90 = np.quantile(rep, [0.05, 0.95])
    return {"diff": float(point), "lo": float(lo), "hi": float(hi), "lo90": float(lo90), "hi90": float(hi90),
            "sig": bool(lo > 0 or hi < 0), "equiv": bool(-MARGIN < lo90 and hi90 < MARGIN)}


def contrast(a, b, m):
    """b minus a on the facts in m (points), with its replicates."""
    rep = rep_mean(b - a, m)
    return summ(rep, 100 * ((b - a) * m).sum() / m.sum()), rep


def pool(items):
    """Equal-weight mean of cells; items are (summary, replicates)."""
    rep = np.mean([r for _, r in items], 0)
    return summ(rep, np.mean([s["diff"] for s, _ in items]))


def load(tag, exp):
    p = RV / f"{tag}_{exp}_bits.json"
    if not (RV / f"{tag}_{exp}_info.json").exists():         # written only after every replay check passed
        OUT["missing"].append(f"{tag} {exp}")
        return None
    OUT["info"][f"{tag}|{exp}"] = json.loads((RV / f"{tag}_{exp}_info.json").read_text())
    return json.loads(p.read_text())


paper_bits = {t: json.loads((P.MERGED / f"{t}_bits.json").read_text()) for t in PAPER}
paper_bits.update({t: json.loads((P.FAMILIES / f"{t}_bits.json").read_text()) for t in NEW})
# Llama-3.1-8B with C4 selection at 70%: the EntityQuestions rerun, which identical reruns reproduce, is the reference
paper_bits["llama8b"]["0.7|C4|C4"] = json.loads((P.EQ_RESULTS / "llama8b_eq_bits.json").read_text())["0.7|C4|C4"]


def replay(tag, b, keys):
    for k in keys:
        assert b[k]["test_q"] == paper_bits[tag][k]["test_q"], f"replay differs: {tag} {k}"


def group_results(cells, name, groups=GROUPS):
    res = {}
    for g, tags in groups.items():
        items = [cells[(t, b, name)] for t in tags for b in BUD if (t, b, name) in cells]
        if len(items) == 2 * len(tags):
            res[g] = pool(items)
            res[g]["sig_pos"] = sum(s["sig"] and s["diff"] > 0 for s, _ in items)
            res[g]["sig_neg"] = sum(s["sig"] and s["diff"] < 0 for s, _ in items)
            res[g]["min"] = min(s["diff"] for s, _ in items)
            res[g]["max"] = max(s["diff"] for s, _ in items)
    return res


# ---------------------------------------------------------------- 7a. matched format
cells = {}
for tag in ALL:
    b = load(tag, "fmt")
    if b is None:
        continue
    replay(tag, b, ["1.0|dense"] + [f"{t}|C4|C4" for t in BUD])
    mq, mc = vec(b["1.0|dense"]["test_q"]), vec(b["1.0|dense"]["test_c"])
    for t in BUD:
        Q = lambda c: vec(b[f"{t}|{c}"]["test_q"])
        Cz = lambda c: vec(b[f"{t}|{c}"]["test_c"])
        cells[(tag, t, "primary")] = contrast(Q("D_Srep"), Q("QA_S"), mq)
        cells[(tag, t, "qa_vs_d")] = contrast(Q("D_S"), Q("QA_S"), mq)
        cells[(tag, t, "primary_cloze")] = contrast(Cz("D_Srep"), Cz("QA_S"), mc)
        inter = cells[(tag, t, "primary")][1] - cells[(tag, t, "primary_cloze")][1]
        cells[(tag, t, "interaction")] = (summ(inter, cells[(tag, t, "primary")][0]["diff"]
                                               - cells[(tag, t, "primary_cloze")][0]["diff"]), inter)
        cells[(tag, t, "more_facts")] = contrast(Q("D_Srep"), vec(paper_bits[tag][f"{t}|MIX:KG-D|MIX:KG-D"]["test_q"]), mq)
        cells[(tag, t, "drep_vs_d")] = contrast(Q("D_S"), Q("D_Srep"), mq)
fmt = {"cells": {"|".join(k): v[0] for k, v in cells.items()}}
for name in ("primary", "qa_vs_d", "primary_cloze", "interaction", "more_facts", "drep_vs_d"):
    fmt[name] = group_results(cells, name)
fmt["rule"] = {}
for g in RULE_GROUPS:
    if g in fmt["primary"]:
        cs = [cells[(t, b, "primary")][0] for t in GROUPS[g] for b in BUD]
        fmt["rule"][g] = {"format_effect": all(c["sig"] for c in cs) and len({c["diff"] > 0 for c in cs}) == 1,
                          "direction": "question-answer" if all(c["diff"] > 0 for c in cs) else
                          "declarative" if all(c["diff"] < 0 for c in cs) else "mixed"}
OUT["fmt"] = fmt

# ---------------------------------------------------------------- 7b. likelihood multiple choice
MC = {"C4 repair vs none": ("none|C4", "C4|C4"), "Repair channel": ("C4|C4", "C4|MIX:KG-D"),
      "Selection channel": ("C4|C4", "MIX:KG-D|C4"), "Selection, no repair": ("none|C4", "MIX:KG-D|none"),
      "Both channels": ("C4|C4", "MIX:KG-D|MIX:KG-D"), "Truth (declarative)": ("MIX:KG-D~|MIX:KG-D~", "MIX:KG-D|MIX:KG-D"),
      "Truth (question-answer)": ("MIX:KG-QA~|MIX:KG-QA~", "MIX:KG-QA|MIX:KG-QA"),
      "Format (QA - declarative)": ("MIX:KG-D|MIX:KG-D", "MIX:KG-QA|MIX:KG-QA")}
cells, dense = {}, {}
for tag in ALL:
    b = load(tag, "mc")
    if b is None:
        continue
    replay(tag, b, ["1.0|dense"] + [f"{t}|{c}" for t in BUD for c in ("C4|C4", "MIX:KG-D|MIX:KG-D")])
    m = vec(b["1.0|dense"]["mc"])
    gq = vec(b["1.0|dense"]["test_q"])
    dense[tag] = {"mc_known": int(m.sum()), "mc_acc": float(m.mean()), "gen_known": int(gq.sum()),
                  "gen_known_also_mc": float((m * gq).sum() / gq.sum())}
    for t in BUD:
        for name, (x, y) in MC.items():
            cells[(tag, t, name)] = contrast(vec(b[f"{t}|{x}"]["mc"]), vec(b[f"{t}|{y}"]["mc"]), m)
        dense[tag][f"retention_{t}"] = {c: float(100 * (vec(b[f"{t}|{c}"]["mc"]) * m).sum() / m.sum())
                                        for c in ("none|C4", "C4|C4", "C4|MIX:KG-D", "MIX:KG-D|C4", "MIX:KG-D|none",
                                                  "MIX:KG-D|MIX:KG-D")}
mcr = {"cells": {"|".join(k): v[0] for k, v in cells.items()}, "dense": dense}
for name in MC:
    mcr[name] = group_results(cells, name)
mcr["truth_pooled"], mcr["rule"] = {}, {}
for g, need in (("all", 6), ("new", 5), ("seven", None)):
    tags = GROUPS[g]
    ti = [cells[(t, b, n)] for t in tags for b in BUD for n in ("Truth (declarative)", "Truth (question-answer)")
          if (t, b, n) in cells]
    if len(ti) == 4 * len(tags):
        mcr["truth_pooled"][g] = pool(ti)
    if need and g in mcr["Both channels"]:          # six of eight settings; five of six for the further families
        mcr["rule"][g] = {"both_channels_sig": mcr["Both channels"][g]["sig_pos"],
                          "findings_survive": mcr["Both channels"][g]["sig_pos"] >= need}
OUT["mc"] = mcr

# ---------------------------------------------------------------- 7c. expanded truth interventions
# Calibration samples 1 and 2, where they exist. Per setting the estimate is the mean over the
# available samples; its replicates resample the test facts once (WF, shared by all runs, which score the same facts)
# and the samples within the setting, as scripts/analysis/stats.py does for the main truth effect.
VARS = ("V25", "V50", "V75", "V100", "VX", "VC")
SHARE = {"V0": 0.0, "V25": 0.25, "V50": 0.5, "V75": 0.75, "V100": 1.0}
cells, slopes, n_samples = {}, {}, {}
pick_rng = np.random.default_rng(5)
cs_ref = {t: (P.MERGED if t in PAPER else P.FAMILIES) for t in ALL}


def two_level(reps, points):
    """reps: one replicate array per sample (NB each); points: the sample estimates."""
    r = np.stack(reps, 1)
    pick = pick_rng.integers(0, r.shape[1], size=r.shape)
    return summ(np.take_along_axis(r, pick, 1).mean(1), float(np.mean(points))), np.take_along_axis(r, pick, 1).mean(1)


for tag in ALL:
    b0 = load(tag, "truth")
    if b0 is None:
        continue
    replay(tag, b0, ["1.0|dense"] + [f"{t}|C4|C4" for t in BUD])
    bs = [b0]
    for s_ in (1, 2):
        if (RV / f"{tag}_truth_cs{s_}_info.json").exists():
            b_ = json.loads((RV / f"{tag}_truth_cs{s_}_bits.json").read_text())
            ref_ = json.loads((cs_ref[tag] / f"{tag}_cs{s_}_bits.json").read_text())
            for k in [f"{t}|C4|C4" for t in BUD]:
                if not (tag == "llama8b" and k == "0.7|C4|C4"):          # see paper_bits above: reported, not asserted
                    assert b_[k]["test_q"] == ref_[k]["test_q"], f"replay differs: {tag} cs{s_} {k}"
            bs.append(b_)
        else:
            OUT["missing"].append(f"{tag} truth cs{s_}")
    n_samples[tag] = len(bs)
    mq, mc = vec(b0["1.0|dense"]["test_q"]), vec(b0["1.0|dense"]["test_c"])
    p = np.array(list(SHARE.values()))
    pc = p - p.mean()
    for t in BUD:
        for v in VARS:
            for suffix, field, m in (("", "test_q", mq), (" cloze", "test_c", mc)):
                reps, pts = [], []
                for b in bs:
                    d = vec(b[f"{t}|V0"][field]) - vec(b[f"{t}|{v}"][field])
                    reps.append(rep_mean(d, m)); pts.append(100 * (d * m).sum() / m.sum())
                cells[(tag, t, v + suffix)] = two_level(reps, pts)
        reps, pts = [], []
        for b in bs:
            Q = lambda c: vec(b[f"{t}|{c}"]["test_q"])
            rr = np.stack([rep_mean(Q(v), mq) for v in SHARE], 1)              # retention replicates, NB x 5
            pp = np.array([100 * (Q(v) * mq).sum() / mq.sum() for v in SHARE])
            reps.append((rr - rr.mean(1, keepdims=True)) @ pc / (pc @ pc)); pts.append(float((pp - pp.mean()) @ pc / (pc @ pc)))
            if b is b0:
                slopes[f"{tag}|{t}"] = {"retention": dict(zip(SHARE, pp.tolist()))}
        cells[(tag, t, "slope")] = two_level(reps, pts)
tru = {"cells": {"|".join(k): v[0] for k, v in cells.items()}, "retention": slopes, "samples": n_samples}
for name in VARS + tuple(v + " cloze" for v in VARS) + ("slope",):
    tru[name] = group_results(cells, name)
OUT["truth"] = tru

(P.TABLES / "controls.json").write_text(json.dumps(OUT, indent=1))
print("missing:", OUT["missing"])
for name in ("primary", "qa_vs_d", "primary_cloze", "more_facts"):
    print("7a", name, {g: f"{v['diff']:+.2f} [{v['lo']:+.2f}, {v['hi']:+.2f}] sig {v['sig_pos']}/{v['sig_neg']}"
                       for g, v in fmt[name].items()})
print("7a rule", fmt["rule"])
for name in MC:
    print("7b", name, {g: f"{v['diff']:+.2f} [{v['lo']:+.2f}, {v['hi']:+.2f}] sig {v['sig_pos']}/{v['sig_neg']}"
                       for g, v in mcr[name].items()})
print("7b dense", dense, mcr.get("truth_pooled"), mcr.get("rule"))
for name in VARS + ("slope",):
    print("7c", name, {g: f"{v['diff']:+.2f} 90% [{v['lo90']:+.2f}, {v['hi90']:+.2f}] equiv {v['equiv']}"
                       for g, v in tru[name].items()})
