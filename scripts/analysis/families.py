"""The analysis of paper_models.py on the three further families, with Qwen2.5 and OLMo-2 computed by the same code.

python scripts/analysis/families.py  ->  outputs/tables/summary.json, outputs/reports/families.md, outputs/figures/ext_*
Reads results/families (further families) and results/merged and results/width_check (Qwen2.5 and OLMo-2). Every
contrast, test and definition follows paper_models.py. Sections whose result files do not exist yet are skipped.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from kgcal import paths as P            # noqa: E402
from kgcal import data as D            # noqa: E402  (PopQA loader and split, read-only)
from kgcal import model_ops as MO      # noqa: E402  (width rule, read-only)

NEWRES = P.FAMILIES
OLDRES, OLDWIDTH = P.MERGED, P.WIDTHS
TAB, FIG = P.TABLES, P.FIGURES
NEW = [("llama8b", "Llama-3.1-8B"), ("mistral7b", "Mistral-7B-v0.3"), ("phi4", "Phi-4")]
OLD = [("qwen15b", "Qwen2.5-1.5B"), ("olmo1b", "OLMo-2-1B"), ("qwen7b", "Qwen2.5-7B"), ("olmo7b", "OLMo-2-7B")]
LABEL = dict(NEW + OLD)
BUDGETS = (0.7, 0.5)
V1 = ("KG-O", "KG-A")
K = {"none": "none|C4", "bias": "bias|C4", "c4": "C4|C4", "rep": "C4|MIX:KG-D", "sel": "MIX:KG-D|C4",
     "selnone": "MIX:KG-D|none", "both": "MIX:KG-D|MIX:KG-D", "qa": "MIX:KG-QA|MIX:KG-QA",
     "dcf": "MIX:KG-D~|MIX:KG-D~", "qacf": "MIX:KG-QA~|MIX:KG-QA~", "syn": "MIX:KG-S|MIX:KG-S",
     "wiki": "MIX:WIKI|MIX:WIKI", "other": "MIX:KG-O2|MIX:KG-O2", "anch": "MIX:KG-Ac|MIX:KG-Ac"}
# name -> (a, b): the difference is retention(b) - retention(a), as contrast() in paper_models.py
CONTRASTS = {"CfourRep": ("none", "c4"), "RepChan": ("c4", "rep"), "SelChan": ("c4", "sel"),
             "SelNoRep": ("none", "selnone"), "Both": ("c4", "both"), "TruthDecl": ("dcf", "both"),
             "TruthQA": ("qacf", "qa"), "Format": ("both", "qa"), "Ent": ("syn", "both"), "Rel": ("other", "both"),
             "Wiki": ("wiki", "both"), "Anch": ("both", "anch")}
COLORS = {"C4": "#7f7f7f", "WIKI": "#8c564b", "KG-D": "#1f77b4", "KG-D~": "#ff7f0e", "KG-QA": "#2ca02c",
          "KG-QA~": "#98df8a", "KG-S": "#9467bd", "KG-Ac": "#d62728", "KG-O2": "#17becf"}


# ---------------------------------------------------------------- helpers (as in paper_models.py)
def load(d: Path, stem: str):
    p, b = d / f"{stem}.jsonl", d / f"{stem}_bits.json"
    if not (p.exists() and b.exists()):
        return None, None
    recs = {}
    for line in open(p):
        r = json.loads(line)
        recs[f"{r['target']}|{r['cond']}"] = r
    bits = {k: {f: np.array([c == "1" for c in v[f]], bool) for f in v} for k, v in json.loads(b.read_text()).items()}
    return recs, bits


def paired(a, b, mask, n_boot=10000, seed=0):
    """Mean of b minus a over facts in mask, paired bootstrap 95% CI, McNemar exact p."""
    x, y = a[mask].astype(float), b[mask].astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    diffs = (y[idx] - x[idx]).mean(1)
    up, down = int(((~a) & b & mask).sum()), int((a & (~b) & mask).sum())
    p = binomtest(up, up + down, 0.5).pvalue if up + down else 1.0
    d = {"diff": float(y.mean() - x.mean()), "lo": float(np.quantile(diffs, 0.025)),
         "hi": float(np.quantile(diffs, 0.975)), "p": float(p)}
    d["sig"] = bool(d["lo"] > 0 or d["hi"] < 0)
    return d


def ret(bits, key, dense):
    return float(bits[key]["test_q"][dense].mean())


def has(bits, t, *cs):
    return bits is not None and all(f"{t}|{K[c]}" in bits for c in cs)


rows = D.load_popqa()
split = D.make_split(rows)
TEST = D.facts_by_split(rows, split)["test"]
POP = np.array([r["s_pop"] for r in TEST])
PROPS = np.array([r["prop"] for r in TEST])
GUESS = {"occupation", "country", "sport", "color", "religion", "genre"}
WCO = np.array(json.loads((P.SPLITS / "leakage_flags.json").read_text())["wiki_cooccur"], bool)


# ---------------------------------------------------------------- per-model summary
def summarize(tag, new: bool):
    R = NEWRES if new else OLDRES
    recs, bits = load(R, tag)
    if bits is None or "1.0|dense" not in bits:
        return None
    dense = bits["1.0|dense"]["test_q"]
    S = {"tag": tag, "label": LABEL[tag], "known": int(dense.sum()), "dense_acc": float(dense.mean()),
         "dense_ppl": recs["1.0|dense"]["ppl_wikitext2"], "dense_zs": {k: recs["1.0|dense"].get(k) for k in ("arc_easy", "hellaswag")},
         "settings": {}}
    seeds = {0: (recs, bits)}
    for s in (1, 2):
        r_, b_ = load(R, f"{tag}_cs{s}")
        if b_ is not None:
            seeds[s] = (r_, b_)
    zs_recs = recs if (new or tag in ("qwen7b", "olmo7b")) else load(R, f"{tag}_zs")[0]
    for t in BUDGETS:
        if not has(bits, t, "none", "c4"):
            continue
        st = {"ret": {c: 100 * ret(bits, f"{t}|{K[c]}", dense) for c in K if f"{t}|{K[c]}" in bits},
              "ppl": {c: recs[f"{t}|{K[c]}"]["ppl_wikitext2"] for c in K if f"{t}|{K[c]}" in recs},
              "param_keep": recs[f"{t}|{K['c4']}"]["param_keep"], "ffn_keep": recs[f"{t}|{K['c4']}"]["ffn_keep"]}
        st["contrasts"] = {n: paired(bits[f"{t}|{K[a]}"]["test_q"], bits[f"{t}|{K[b]}"]["test_q"], dense)
                           for n, (a, b) in CONTRASTS.items() if has(bits, t, a, b)}
        if has(bits, t, "wiki", "both"):
            st["contrasts"]["WikiClean"] = paired(bits[f"{t}|{K['wiki']}"]["test_q"], bits[f"{t}|{K['both']}"]["test_q"],
                                                  dense & ~WCO)
            st["wiki_removed"] = int((dense & WCO).sum())
        pd_, pn, pc = S["dense_ppl"], st["ppl"]["none"], st["ppl"]["c4"]
        rn, rc = st["ret"]["none"] / 100, st["ret"]["c4"] / 100
        st["teaser"] = {"ppl_gap_recovered": (pn - pc) / (pn - pd_), "facts_recovered": (rc - rn) / (1 - rn)}
        if "both" in st["ret"]:
            st["teaser"]["facts_recovered_kg"] = (st["ret"]["both"] / 100 - rn) / (1 - rn)
        # perplexity versus retention over repaired activation-norm configurations (no v1 sources)
        pts, allpts = [], []
        for k, r in recs.items():
            if not k.startswith(f"{t}|") or r.get("sel_rule") != "actnorm":
                continue
            src = (r.get("rep_src") or r.get("sel_src") or "").replace("MIX:", "")
            if src in V1:
                continue
            p_ = (r["ppl_wikitext2"], 100 * ret(bits, k, dense), r["cond"])
            allpts.append(p_)
            if r.get("rep") != "none" and (r.get("rep_src") or "").replace("MIX:", "") not in V1:
                pts.append(p_)
        if len(pts) >= 4:
            st["rho"] = float(spearmanr([-p[0] for p in pts], [p[1] for p in pts]).statistic)
            st["lowest_ppl"] = min(pts, key=lambda p: p[0])[2]
            st["best_ret"] = max(allpts, key=lambda p: p[1])[2]
        cand = [(100 * ret(bits, k, dense), r["cond"]) for k, r in recs.items() if k.startswith(f"{t}|")
                and (r.get("rep_src") or r.get("sel_src") or "").replace("MIX:", "") not in V1 and r["cond"] != "dense"]
        st["best_any"] = max(cand)
        # seeds
        sd = {}
        for s, (_, b_) in seeds.items():
            if has(b_, t, "c4", "both"):
                e = {"both_minus_c4": 100 * (ret(b_, f"{t}|{K['both']}", dense) - ret(b_, f"{t}|{K['c4']}", dense)),
                     "c4": 100 * ret(b_, f"{t}|{K['c4']}", dense), "both": 100 * ret(b_, f"{t}|{K['both']}", dense)}
                for n, (cf, tr) in (("TruthDecl", ("dcf", "both")), ("TruthQA", ("qacf", "qa"))):
                    if has(b_, t, cf, tr):
                        e[n] = paired(b_[f"{t}|{K[cf]}"]["test_q"], b_[f"{t}|{K[tr]}"]["test_q"], dense)
                sd[s] = e
        st["seeds"] = sd
        # zero-shot
        if zs_recs is not None and all(f"{t}|{K[c]}" in zs_recs and "hellaswag" in zs_recs[f"{t}|{K[c]}"] for c in ("c4", "both")):
            g = lambda c, m: 100 * zs_recs[f"{t}|{K[c]}"][m]
            st["zs"] = {"hs": g("both", "hellaswag") - g("c4", "hellaswag"), "arc": g("both", "arc_easy") - g("c4", "arc_easy")}
            if f"{t}|{K['rep']}" in zs_recs and "hellaswag" in zs_recs[f"{t}|{K['rep']}"]:
                st["zs"]["hs_rep"] = g("rep", "hellaswag") - g("c4", "hellaswag")
                st["zs"]["arc_rep"] = g("rep", "arc_easy") - g("c4", "arc_easy")
            if "rep" in st["contrasts"] or "RepChan" in st["contrasts"]:
                pass
        # popularity quintiles (70% in the paper)
        q5 = np.quantile(POP, [0.2, 0.4, 0.6, 0.8]); qi = np.searchsorted(q5, POP, side="right")
        m = dense & ~np.isin(PROPS, list(GUESS))
        if has(bits, t, "c4", "both"):
            a_, k_ = bits[f"{t}|{K['c4']}"]["test_q"], bits[f"{t}|{K['both']}"]["test_q"]
            st["pop"] = [(int((m & (qi == q)).sum()), 100 * a_[m & (qi == q)].mean(), 100 * k_[m & (qi == q)].mean())
                         for q in range(5)]
            st["top_share"] = 100 * (m & (qi == 4)).sum() / m.sum()
        S["settings"][t] = st
    # ridge strength and 4x calibration text (70%)
    lam_vals = {}
    if has(bits, 0.7, "c4"):
        lam_vals["1e-4"] = 100 * ret(bits, f"0.7|{K['c4']}", dense)
    for l in ("1e-2", "1e-1"):
        _, b_ = load(R, f"{tag}_lam{l}")
        if has(b_, 0.7, "c4"):
            lam_vals[l] = 100 * ret(b_, f"0.7|{K['c4']}", dense)
    S["lam"] = lam_vals
    r4, b4 = load(R, f"{tag}_cal4x")
    if has(b4, 0.7, "none", "c4"):
        S["cal4x"] = {"d": paired(b4[f"0.7|{K['none']}"]["test_q"], b4[f"0.7|{K['c4']}"]["test_q"], dense),
                      "ppl4": (r4[f"0.7|{K['none']}"]["ppl_wikitext2"], r4[f"0.7|{K['c4']}"]["ppl_wikitext2"]),
                      "ppl1": (recs[f"0.7|{K['none']}"]["ppl_wikitext2"], recs[f"0.7|{K['c4']}"]["ppl_wikitext2"])}
    # Lemma 1 terms
    tp = R / f"{tag}_theory.jsonl"
    if tp.exists():
        T = [json.loads(l) for l in open(tp)]
        S["mech"] = mechanism(T)
    # widths
    S["widths"] = widths_summary(tag, new, dense)
    return S


def mean_ratio(T, t, sel, cons, tgt="question"):
    v = [r["web_heldout_ratio"][cons] if tgt == "web" else r[tgt][cons]["ratio"] for r in T
         if r["target"] == t and r["sel"] == sel and (tgt == "web" or r.get(tgt)) and
         (cons in (r["web_heldout_ratio"] if tgt == "web" else r[tgt]))]
    return float(np.mean(v)) if v else None


def mechanism(T):
    out = {}
    for t in BUDGETS:
        out[t] = {"c4": mean_ratio(T, t, "actnorm:C4", "ridge:C4"), "c4web": mean_ratio(T, t, "actnorm:C4", "ridge:C4", "web"),
                  "mix": mean_ratio(T, t, "actnorm:C4", "ridge:MIX:KG-D"),
                  "kgweb": mean_ratio(T, t, "actnorm:C4", "ridge:KG-D", "web")}
        diffs, tdir = [], []
        for tr, cf in (("MIX:KG-D", "MIX:KG-D~"), ("MIX:KG-QA", "MIX:KG-QA~")):
            a_ = mean_ratio(T, t, f"actnorm:{tr}", f"ridge:{tr}"); b_ = mean_ratio(T, t, f"actnorm:{cf}", f"ridge:{cf}")
            if a_ is not None and b_ is not None:
                diffs.append(abs(a_ - b_)); tdir.append(a_ < b_)
        out[t]["truth_absdiff"] = max(diffs) if diffs else None
        out[t]["truth_true_lower"] = tdir
        w = [r["question"]["ridge:C4"]["ratio"] > 1 for r in T if r["target"] == t and r["sel"] == "actnorm:C4" and r.get("question")]
        out[t]["c4_worse_blocks"] = (int(sum(w)), len(w))
    gaps = [x["identity_gap"] for r in T for f in ("question", "cloze") if r.get(f) for c, x in r[f].items() if isinstance(x, dict)]
    out["identity_gap_max"] = max(gaps) if gaps else None
    return out


def widths_summary(tag, new, dense):
    if new:
        p = NEWRES / f"{tag}_dims.json"
        if not p.exists():
            return None
        Rd = json.loads(p.read_text())
        N, L = Rd["N"], Rd["L"]
        dc = np.array([e["code"] for e in Rd["layers"]]); dg = np.array([e["gride"] for e in Rd["layers"]])
        out = {"repro_gap": Rd["repro_gap"], "ratio_mean": float((dc / dg).mean()), "ratio_min": float((dc / dg).min()),
               "ratio_max": float((dc / dg).max()), "spearman": float(spearmanr(dc, dg).statistic), "moved": {}}
        for f in BUDGETS:
            wc = np.array(MO.widths(list(dc), MO.alpha_for_fraction(list(dc), N, f), N))
            wg = np.array(MO.widths(list(dg), MO.alpha_for_fraction(list(dg), N, f), N))
            out["moved"][f] = float(np.abs(wc - wg).sum() / 2 / (L * N))
        Dm = np.array([[x["code"] for x in Rd["subsample_seeds"][s]] for s in Rd["subsample_seeds"]])
        w = [np.array(MO.widths(list(Dm[i]), MO.alpha_for_fraction(list(Dm[i]), N, 0.7), N)) for i in range(Dm.shape[0])]
        out["seed_moved"] = float(np.mean([np.abs(w[i] - w[j]).sum() / 2 / (L * N) for i in range(3) for j in range(i + 1, 3)]))
        _, bg = load(NEWRES, f"{tag}_gmle")
    else:
        p = OLDWIDTH / f"widths_{tag}.json"
        if not p.exists():
            return None
        Wd = json.loads(p.read_text())
        out = {"repro_gap": Wd["repro_gap"], "ratio_mean": Wd["ratio_mean"], "ratio_min": Wd["ratio_min"],
               "ratio_max": Wd["ratio_max"], "spearman": Wd["spearman_code_gride"],
               "moved": {f: Wd["budgets"][str(f)]["moved_gride"] for f in BUDGETS}, "seed_moved": Wd["seed_moved_code"]}
        _, bg = load(OLDWIDTH, f"{tag}_gmle")
    if bg is not None:
        out["gmle_both"] = {f: paired(bg[f"{f}|{K['c4']}"]["test_q"], bg[f"{f}|{K['both']}"]["test_q"], dense)
                            for f in BUDGETS if has(bg, f, "c4", "both")}
        out["gmle_c4rep"] = {f: paired(bg[f"{f}|{K['none']}"]["test_q"], bg[f"{f}|{K['c4']}"]["test_q"], dense)
                             for f in BUDGETS if has(bg, f, "none", "c4")}
    return out


SUM = {}
for tag, _ in OLD:
    SUM[tag] = summarize(tag, new=False)
for tag, _ in NEW:
    s = summarize(tag, new=True)
    if s is not None:
        SUM[tag] = s
NEWHAVE = [t for t, _ in NEW if t in SUM]


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    return o


(TAB / "summary.json").write_text(json.dumps(clean(SUM), indent=1))


# ---------------------------------------------------------------- RESULTS.md
def cells(group):
    return [(tag, t) for tag in group if tag in SUM for t in BUDGETS if t in SUM[tag]["settings"]]


def C(tag, t, name):
    return SUM[tag]["settings"][t]["contrasts"].get(name)


def fmt_c(d):
    if d is None:
        return "--"
    v = f"{100 * d['diff']:+.1f}"
    return f"**{v}**" if d["sig"] else v


def rng_txt(vals, sign=True, nd=1):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "--"
    f = (lambda v: f"{v:+.{nd}f}") if sign else (lambda v: f"{v:.{nd}f}")
    return f"{f(min(vals))} to {f(max(vals))}"


def contrast_summary(group, name):
    ds = [C(tag, t, name) for tag, t in cells(group)]
    ds = [d for d in ds if d is not None]
    if not ds:
        return None
    return {"n": len(ds), "range": rng_txt([100 * d["diff"] for d in ds]), "mean": float(np.mean([100 * d["diff"] for d in ds])),
            "pos": sum(d["diff"] > 0 for d in ds), "sig_pos": sum(d["sig"] and d["diff"] > 0 for d in ds),
            "sig_neg": sum(d["sig"] and d["diff"] < 0 for d in ds)}


OLDT, NEWT = [t for t, _ in OLD], NEWHAVE
md = ["# Results of the three further model families (generated by scripts/analysis/families.py; do not edit)", ""]
md += [f"New models with results: {', '.join(LABEL[t] for t in NEWHAVE) or 'none yet'}. Paper models: computed from "
       "results/merged with the same code. Bold contrasts: 95% paired-bootstrap interval excludes zero. "
       "All retention values are on the test facts the dense model answers (dense-known facts).", ""]

# dense models
md += ["## Dense models", "", "| model | dense-known test facts | test accuracy | WikiText-2 ppl | ARC-Easy | HellaSwag |",
       "|---|---|---|---|---|---|"]
pct = lambda v: "--" if v is None else f"{100 * v:.1f}"
for tag in OLDT + NEWT:
    s = SUM[tag]
    z = s["dense_zs"]
    md.append(f"| {s['label']} | {s['known']:,} | {100 * s['dense_acc']:.1f}% | {s['dense_ppl']:.2f} | "
              f"{pct(z['arc_easy'])} | {pct(z['hellaswag'])} |")
md.append("")

# main retention table (Table 2 of the paper) for the new models
ROWS = [("none", "C4 / none"), ("bias", "C4 / bias"), ("c4", "C4 / C4"), ("rep", "C4 / KG"), ("sel", "KG / C4"),
        ("selnone", "KG / none"), ("both", "KG / KG"), ("qa", "KG-QA / KG-QA"), ("dcf", "KG-D~ / KG-D~"),
        ("qacf", "KG-QA~ / KG-QA~"), ("syn", "KG-S / KG-S"), ("other", "KG-O / KG-O"), ("wiki", "WIKI / WIKI"),
        ("anch", "KG-A / KG-A")]
for title, group in (("New models", NEWT), ("Paper models (for comparison)", OLDT)):
    cs = cells(group)
    if not cs:
        continue
    md += [f"## Fact retention (%), {title.lower()}", "",
           "| selection / repair | " + " | ".join(f"{LABEL[tag]} {int(100 * t)}%" for tag, t in cs) + " |",
           "|---|" + "---|" * len(cs)]
    for c, name in ROWS:
        md.append(f"| {name} | " + " | ".join(f"{SUM[tag]['settings'][t]['ret'][c]:.1f}" if c in SUM[tag]["settings"][t]["ret"]
                                               else "--" for tag, t in cs) + " |")
    md.append("| WikiText-2 ppl, C4 / none -> C4 / C4 | " + " | ".join(
        f"{SUM[tag]['settings'][t]['ppl']['none']:.1f} -> {SUM[tag]['settings'][t]['ppl']['c4']:.1f}" for tag, t in cs) + " |")
    md.append("| whole-model parameters kept | " + " | ".join(f"{100 * SUM[tag]['settings'][t]['param_keep']:.1f}%" for tag, t in cs) + " |")
    md.append("")

# contrasts table (Table 3 and Figure 4 of the paper)
NAMES = [("CfourRep", "C4 repair vs none"), ("RepChan", "repair channel (C4 / KG vs C4 / C4)"),
         ("SelChan", "selection channel (KG / C4 vs C4 / C4)"), ("SelNoRep", "selection, no repair"),
         ("Both", "both channels (KG / KG vs C4 / C4)"), ("TruthDecl", "truth, declarative (true - counterfactual)"),
         ("TruthQA", "truth, question-answer (true - counterfactual)"), ("Format", "format (question-answer - declarative)"),
         ("Ent", "entities (real - made-up subjects)"), ("Rel", "relations (tested - other)"),
         ("Wiki", "text (KG statements - Wikipedia)"), ("WikiClean", "text, without Wikipedia co-occurrence facts"),
         ("Anch", "anchoring (anchored - entity-disjoint)")]
cs = cells(NEWT)
if cs:
    md += ["## Paired contrasts, new models (retention points)", "",
           "| contrast | " + " | ".join(f"{LABEL[tag]} {int(100 * t)}%" for tag, t in cs) + " |", "|---|" + "---|" * len(cs)]
    for n, lab in NAMES:
        md.append(f"| {lab} | " + " | ".join(fmt_c(C(tag, t, n)) for tag, t in cs) + " |")
    md.append("")

# claim-by-claim comparison
md += ["## The paper's claims, paper models versus new models", "",
       "Ranges are over model x budget settings; counts are significant differences (95% interval excludes zero).", "",
       "| claim | paper models | new models |", "|---|---|---|"]


def side(fn):
    return fn(OLDT), (fn(NEWT) if cells(NEWT) else "--")


def teaser_txt(group):
    cs = cells(group)
    if not cs:
        return "--"
    ts = [SUM[tag]["settings"][t]["teaser"] for tag, t in cs]
    g = np.mean([x["ppl_gap_recovered"] for x in ts]); f = np.mean([x["facts_recovered"] for x in ts])
    k = [x["facts_recovered_kg"] for x in ts if "facts_recovered_kg" in x]
    return (f"C4 repair recovers {100 * g:.0f}% of the perplexity gap but {100 * f:.0f}% of the lost facts "
            f"(KG mixture {100 * np.mean(k):.0f}%), mean of {len(cs)}")


def cstat(name, what="range"):
    def f(group):
        s = contrast_summary(group, name)
        if s is None:
            return "--"
        return f"{s['range']}; {s['pos']}/{s['n']} positive; sig. + {s['sig_pos']}, sig. - {s['sig_neg']}"
    return f


def rho_txt(group):
    cs = [(tag, t) for tag, t in cells(group) if "rho" in SUM[tag]["settings"][t]]
    if not cs:
        return "--"
    r = [SUM[tag]["settings"][t]["rho"] for tag, t in cs]
    mism = sum(SUM[tag]["settings"][t]["lowest_ppl"] != SUM[tag]["settings"][t]["best_ret"] for tag, t in cs)
    return f"Spearman {rng_txt(r, nd=2)}; lowest-perplexity configuration is not the best for facts in {mism}/{len(cs)}"


def best_txt(group):
    cs = cells(group)
    if not cs:
        return "--"
    from collections import Counter
    c = Counter(SUM[tag]["settings"][t]["best_any"][1] for tag, t in cs)
    return "; ".join(f"{k} x{v}" for k, v in c.most_common())


def seeds_txt(group):
    v = [e["both_minus_c4"] for tag, t in cells(group) for s, e in SUM[tag]["settings"][t]["seeds"].items()]
    if not v:
        return "--"
    return f"KG / KG minus C4 / C4 over {len(v)} runs: {rng_txt(v)}; positive in {sum(x > 0 for x in v)}/{len(v)}"


def truth_seed_txt(group):
    ds = [e[n] for tag, t in cells(group) for s, e in SUM[tag]["settings"][t]["seeds"].items() for n in ("TruthDecl", "TruthQA") if n in e]
    if not ds:
        return "--"
    v = [100 * d["diff"] for d in ds]
    return (f"{len(ds)} differences {rng_txt(v)}, mean {np.mean(v):+.1f}; sig. for true {sum(d['sig'] and d['diff'] > 0 for d in ds)}, "
            f"for counterfactual {sum(d['sig'] and d['diff'] < 0 for d in ds)}")


def zs_txt(group):
    z = [SUM[tag]["settings"][t]["zs"] for tag, t in cells(group) if "zs" in SUM[tag]["settings"][t]]
    if not z:
        return "--"
    s = f"HellaSwag {rng_txt([x['hs'] for x in z])}, ARC-Easy {rng_txt([x['arc'] for x in z])} (KG / KG minus C4 / C4)"
    r = [x["hs_rep"] for x in z if "hs_rep" in x]
    return s + (f"; repair-only HellaSwag {rng_txt(r)}" if r else "")


def lam_txt(group):
    out = []
    for tag in group:
        if tag in SUM and len(SUM[tag]["lam"]) == 3:
            l = SUM[tag]["lam"]
            out.append(f"{LABEL[tag]} {l['1e-4']:.1f} / {l['1e-2']:.1f} / {l['1e-1']:.1f}")
    return "; ".join(out) if out else "--"


def cal4_txt(group):
    out = []
    for tag in group:
        if tag in SUM and "cal4x" in SUM[tag]:
            c = SUM[tag]["cal4x"]
            out.append(f"{LABEL[tag]} {100 * c['d']['diff']:+.1f}{'*' if c['d']['sig'] else ''} "
                       f"(ppl cost {c['ppl1'][1] - c['ppl1'][0]:+.2f} -> {c['ppl4'][1] - c['ppl4'][0]:+.2f})")
    return "; ".join(out) if out else "--"


def mech_txt(key, nd=2):
    def f(group):
        v = [SUM[tag]["mech"][t][key] for tag in group if tag in SUM and "mech" in SUM[tag] for t in BUDGETS
             if SUM[tag]["mech"][t][key] is not None]
        return rng_txt(v, sign=False, nd=nd) if v else "--"
    return f


def mech_truth(group):
    d = [(SUM[tag]["mech"][t]["truth_absdiff"], SUM[tag]["mech"][t]["truth_true_lower"]) for tag in group
         if tag in SUM and "mech" in SUM[tag] for t in BUDGETS if SUM[tag]["mech"][t]["truth_absdiff"] is not None]
    if not d:
        return "--"
    low = [x for _, l in d for x in l]
    return f"max |true - counterfactual| {max(x for x, _ in d):.3f}; true lower in {sum(low)}/{len(low)}"


def gap_txt(group):
    g = [SUM[tag]["mech"]["identity_gap_max"] for tag in group if tag in SUM and "mech" in SUM[tag]]
    return f"largest {max(g):.1e}" if g else "--"


def width_txt(group):
    w = [SUM[tag]["widths"] for tag in group if tag in SUM and SUM[tag].get("widths")]
    if not w:
        return "--"
    s = (f"code / GRIDE-MLE {rng_txt([x['ratio_mean'] for x in w], sign=False, nd=2)}; GRIDE-MLE widths move "
         f"{rng_txt([100 * v for x in w for v in x['moved'].values()], sign=False)}% of units (subsample redraw "
         f"{rng_txt([100 * x['seed_moved'] for x in w], sign=False)}%)")
    g = [d for x in w for d in x.get("gmle_both", {}).values()]
    if g:
        s += f"; both channels under GRIDE-MLE widths {rng_txt([100 * d['diff'] for d in g])}, sig. + {sum(d['sig'] and d['diff'] > 0 for d in g)}/{len(g)}"
    return s


def pop_txt(group):
    up = same = n = 0
    for tag in group:
        if tag in SUM and 0.7 in SUM[tag]["settings"] and "pop" in SUM[tag]["settings"][0.7]:
            for _, x, y in SUM[tag]["settings"][0.7]["pop"]:
                up += round(y) > round(x); same += round(y) == round(x); n += 1
    return f"KG / KG above C4 / C4 in {up}/{n} model-quintile cells at 70%, equal in {same}" if n else "--"


CLAIMS = [("R1 generic repair restores perplexity, not facts", teaser_txt),
          ("R2 C4 repair vs no repair", cstat("CfourRep")),
          ("R3 perplexity ranks configurations differently", rho_txt),
          ("R4 repair channel", cstat("RepChan")), ("R5 selection channel", cstat("SelChan")),
          ("R5b selection without repair", cstat("SelNoRep")),
          ("R6 both channels", cstat("Both")), ("R6b both channels over calibration samples", seeds_txt),
          ("R7 best configuration per setting", best_txt),
          ("R8a C4 / C4 retention at ridge 1e-4 / 1e-2 / 1e-1 (70%)", lam_txt),
          ("R8b 4x C4: repair effect on retention (70%)", cal4_txt),
          ("R9 generic cost of KG calibration (zero-shot)", zs_txt),
          ("F1a truth, declarative (sample 0)", cstat("TruthDecl")), ("F1b truth, question-answer (sample 0)", cstat("TruthQA")),
          ("F2 truth over the three samples", truth_seed_txt),
          ("F4 format: question-answer minus declarative", cstat("Format")),
          ("F5 relations: tested minus other", cstat("Rel")), ("F6 KG statements minus Wikipedia", cstat("Wiki")),
          ("F6b same, without co-occurrence facts", cstat("WikiClean")),
          ("F6c entities: real minus made-up subjects", cstat("Ent")),
          ("F7 anchoring", cstat("Anch")), ("P1 popularity (70%)", pop_txt),
          ("T1 Lemma 1 identity gap", gap_txt),
          ("T2 C4 / C4 error ratio on knowledge-question contexts (1 = no repair)", mech_txt("c4")),
          ("T2b C4 / C4 error ratio on held-out Web text", mech_txt("c4web")),
          ("T3 repair fitted on KG text only, ratio on held-out Web text", mech_txt("kgweb")),
          ("T4 C4 / MIX(KG-D) ratio on knowledge-question contexts", mech_txt("mix")),
          ("T5 truth acts only through moments", mech_truth),
          ("Width estimator", width_txt)]
for name, fn in CLAIMS:
    a, b = side(fn)
    md.append(f"| {name} | {a} | {b} |")
md.append("")

# per-setting truth over seeds and seed spread, new models
cs = cells(NEWT)
if cs and any(SUM[tag]["settings"][t]["seeds"].get(1) for tag, t in cs):
    md += ["## Calibration samples, new models", "",
           "| setting | C4 / C4 (s0, s1, s2) | KG / KG (s0, s1, s2) | truth decl. (s0, s1, s2) | truth QA (s0, s1, s2) |",
           "|---|---|---|---|---|"]
    for tag, t in cs:
        sd = SUM[tag]["settings"][t]["seeds"]
        g = lambda k: ", ".join(f"{sd[s][k]:.1f}" if s in sd else "--" for s in (0, 1, 2))
        h = lambda k: ", ".join(fmt_c(sd[s].get(k)) if s in sd else "--" for s in (0, 1, 2))
        md.append(f"| {LABEL[tag]} {int(100 * t)}% | {g('c4')} | {g('both')} | {h('TruthDecl')} | {h('TruthQA')} |")
    md.append("")

(P.REPORTS / "families.md").write_text("\n".join(md) + "\n")

# ---------------------------------------------------------------- figures (as Figures 2 and 4 of the paper)
if cells(NEWT):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False})
    short = lambda m: m.replace("-v0.3", "").replace("Llama-3.1-", "Llama ").replace("Mistral-", "Mistral ")
    PROPERTY = [("TruthDecl", "Truth: true minus counterfactual (declarative)"),
                ("TruthQA", "Truth: true minus counterfactual (question-answer)"),
                ("Format", "Format: question-answer minus declarative"), ("Ent", "Entities: real minus made-up subjects"),
                ("Rel", "Relations: tested minus other relations"), ("Wiki", "Text: KG statements minus Wikipedia"),
                ("Anch", "Anchoring: anchored minus entity-disjoint")]
    mk = {"llama8b": "o", "mistral7b": "s", "phi4": "^"}
    col = {("llama8b", 0.7): "#08519c", ("llama8b", 0.5): "#6baed6", ("mistral7b", 0.7): "#a50f15", ("mistral7b", 0.5): "#fb6a4a",
           ("phi4", 0.7): "#006d2c", ("phi4", 0.5): "#74c476"}
    cs = cells(NEWT)
    fig, ax = plt.subplots(figsize=(3.4, 4.3))
    n = len(PROPERTY)
    for ri, (name, lab) in enumerate(PROPERTY):
        y0 = n - 1 - ri
        for ci, (tag, t) in enumerate(cs):
            d = C(tag, t, name)
            if d is None:
                continue
            y = y0 + 0.32 - ci * (0.64 / max(1, len(cs) - 1))
            ax.plot([100 * d["lo"], 100 * d["hi"]], [y, y], color=col[(tag, t)], lw=0.9)
            ax.plot(100 * d["diff"], y, marker=mk[tag], color=col[(tag, t)], ms=3.2, ls="none")
    ax.axvline(0, color="#777", lw=0.6)
    ax.set_yticks(range(n)); ax.set_yticklabels([p[1].replace(": ", ":\n") for p in PROPERTY][::-1], fontsize=6.3)
    ax.set_xlabel("Retention difference (points)", fontsize=7)
    hs = [plt.Line2D([], [], marker=mk[c[0]], color=col[c], ls="none", ms=3.5, label=f"{short(LABEL[c[0]])} {int(c[1] * 100)}%")
          for c in cs]
    ax.legend(handles=hs, fontsize=6, frameon=False, loc="lower center", bbox_to_anchor=(0.32, 1.0), ncol=2,
              columnspacing=0.8, handletextpad=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "ext_facts_or_formats.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(FIG / "ext_facts_or_formats.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    for budget in BUDGETS:
        group = [tag for tag in NEWT if budget in SUM[tag]["settings"]]
        if not group:
            continue
        fig, axes = plt.subplots(1, len(group), figsize=(1.75 * len(group), 2.15), squeeze=False)
        for ax, tag in zip(axes[0], group):
            recs, bits = load(NEWRES, tag)
            dense = bits["1.0|dense"]["test_q"]
            for k, r in recs.items():
                if not k.startswith(f"{budget}|") or r.get("sel_rule") != "actnorm":
                    continue
                src = (r.get("rep_src") or r.get("sel_src") or "C4").replace("MIX:", "")
                rr = 100 * ret(bits, k, dense)
                ax.scatter(r["ppl_wikitext2"], rr, s=14, color=COLORS.get(src, "#333"), zorder=3,
                           marker="x" if r.get("rep") == "none" else "o")
                if r["cond"] == K["c4"]:
                    ax.annotate("C4/C4", (r["ppl_wikitext2"], rr), xytext=(4, -9), textcoords="offset points", fontsize=6)
            ax.set_title(f"{short(LABEL[tag])}, {int(budget * 100)}% FFN", fontsize=7)
            ax.set_xlabel("WikiText-2 perplexity", fontsize=6.5); ax.tick_params(labelsize=6)
        axes[0][0].set_ylabel("Fact retention (%)", fontsize=7)
        names = [("C4", "C4"), ("KG-D", "KG-D"), ("KG-D~", "KG-D$^{\\sim}$"), ("KG-QA", "KG-QA"), ("KG-QA~", "KG-QA$^{\\sim}$"),
                 ("KG-S", "KG-S"), ("KG-O2", "KG-O"), ("KG-Ac", "KG-A"), ("WIKI", "WIKI")]
        hs = [plt.Line2D([], [], marker="o", color=COLORS[k], ls="none", ms=4, label=lab) for k, lab in names]
        hs += [plt.Line2D([], [], marker="x", color=COLORS["C4"], ls="none", ms=4, label="C4, no repair"),
               plt.Line2D([], [], marker="x", color=COLORS["KG-D"], ls="none", ms=4, label="KG-D, no repair")]
        fig.legend(handles=hs, loc="lower center", ncol=6, fontsize=6, frameon=False, columnspacing=0.9, handletextpad=0.2)
        fig.tight_layout(rect=(0, 0.17, 1, 1))
        fig.savefig(FIG / f"ext_ppl_vs_retention_{int(100 * budget)}.pdf")
        fig.savefig(FIG / f"ext_ppl_vs_retention_{int(100 * budget)}.png", dpi=200)
        plt.close(fig)
print("wrote", P.REPORTS / "families.md", "| new models with results:", NEWHAVE)
