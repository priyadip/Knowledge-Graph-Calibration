"""Numbers, tables and figures that involve all seven models or the experiments beyond the main grid.

Usage:  python scripts/paper/make_numbers.py      (after the scripts of scripts/analysis, see scripts/reproduce_paper.sh)
Reads outputs/tables/summary.json (per-model statistics of all seven models from scripts/analysis/families.py,
computed with the paired bootstrap of paper_models.py, so its Qwen2.5 and OLMo-2 values equal those of
paper_models.py), the bits of results/families and results/merged, outputs/tables/stats.json, entityquestions.json,
mixture.json, overlap.json, validation.json and controls.json, results/mixture, results/sequential, results/obs and
results/answers.
Writes outputs/tables/numbers08.tex (no macro name shared with numbers.tex, so a clash stops the LaTeX build), the
14-setting tables (*7.tex), the appendix tables (tableB_*.tex, table3_summary.tex, tableA_validation.tex),
outputs/figures/fig4_facts_or_formats7.pdf, figA_ppl_vs_retention_new.pdf and outputs/reports/pending.md. A number or
cell whose run is missing prints \\Pending and is listed in pending.md. Every count written in words has a guard at
the end.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from kgcal import paths as P            # noqa: E402

TAB, FIG, NOTES = P.TABLES, P.FIGURES, P.REPORTS
PAPER = [("qwen15b", "Qwen2.5-1.5B"), ("olmo1b", "OLMo-2-1B"), ("qwen7b", "Qwen2.5-7B"), ("olmo7b", "OLMo-2-7B")]
NEW = [("llama8b", "Llama-3.1-8B"), ("mistral7b", "Mistral-7B-v0.3"), ("phi4", "Phi-4")]
MODELS = PAPER + NEW
LABEL = dict(MODELS)
SHORT = {"qwen15b": "Qwen 1.5B", "olmo1b": "OLMo 1B", "qwen7b": "Qwen 7B", "olmo7b": "OLMo 7B",
         "llama8b": "Llama 8B", "mistral7b": "Mistral 7B", "phi4": "Phi-4"}
SMALL = ("qwen15b", "olmo1b")
LARGE = ("qwen7b", "olmo7b", "llama8b", "mistral7b", "phi4")
BUD = ("0.7", "0.5")
CELLS = [(tag, t) for tag, _ in MODELS for t in BUD]
PCELLS = [c for c in CELLS if c[0] in dict(PAPER)]
NCELLS = [c for c in CELLS if c[0] in dict(NEW)]
LCELLS = [c for c in CELLS if c[0] in LARGE]
K = {"none": "none|C4", "c4": "C4|C4", "rep": "C4|MIX:KG-D", "sel": "MIX:KG-D|C4", "selnone": "MIX:KG-D|none",
     "both": "MIX:KG-D|MIX:KG-D", "qa": "MIX:KG-QA|MIX:KG-QA", "dcf": "MIX:KG-D~|MIX:KG-D~", "qacf": "MIX:KG-QA~|MIX:KG-QA~",
     "syn": "MIX:KG-S|MIX:KG-S", "wiki": "MIX:WIKI|MIX:WIKI", "other": "MIX:KG-O2|MIX:KG-O2", "anch": "MIX:KG-Ac|MIX:KG-Ac"}
NUM: dict[str, str] = {}
PENDING: list[str] = []
PEND = "\\Pending{}"


# ---------------------------------------------------------------- helpers (as in scripts/analysis/paper_models.py)
def load(d: Path, stem: str):
    p, b = d / f"{stem}.jsonl", d / f"{stem}_bits.json"
    if not b.exists():
        return None, None
    recs = {}
    if p.exists():
        for line in open(p):
            r = json.loads(line)
            recs[f"{r['target']}|{r['cond']}"] = r
    bits = {k: {f: np.array([c == "1" for c in v[f]], bool) for f in v} for k, v in json.loads(b.read_text()).items()}
    return recs, bits


def paired(a, b, mask, n_boot=10000, seed=0):
    """Mean of b minus a over facts in mask, paired bootstrap 95% CI, McNemar exact p (as in paper_models.py)."""
    x, y = a[mask].astype(float), b[mask].astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    diffs = (y[idx] - x[idx]).mean(1)
    up, down = int(((~a) & b & mask).sum()), int((a & (~b) & mask).sum())
    p = binomtest(up, up + down, 0.5).pvalue if up + down else 1.0
    d = {"diff": float(y.mean() - x.mean()), "lo": float(np.quantile(diffs, 0.025)), "hi": float(np.quantile(diffs, 0.975)),
         "p": float(p)}
    d["sig"] = d["lo"] > 0 or d["hi"] < 0
    return d


def nfmt(v, sign=False, nd=1):
    v = round(float(v), nd)
    if v == 0:
        v = 0.0
    s = f"{v:+,.{nd}f}" if sign else f"{v:,.{nd}f}"
    return f"\\ensuremath{{{s}}}" if sign else s


def num(name, value, sign=False, nd=1):
    assert name not in NUM, name
    NUM[name] = value if isinstance(value, str) else nfmt(value, sign, nd)


def num_range(name, vals, sign=False, nd=1):
    vals = [v for v in vals if v is not None]
    num(name + "Lo", min(vals), sign, nd)
    num(name + "Hi", max(vals), sign, nd)
    return min(vals), max(vals)


WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
         "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"]
words = lambda n: WORDS[n] if 0 <= n < len(WORDS) else f"{n:,}"
pct = lambda x: 100 * x


def pending(item):
    if item not in PENDING:
        PENDING.append(item)
    return PEND


# ---------------------------------------------------------------- data
SUM = json.loads((TAB / "summary.json").read_text())
ST = lambda tag, t: SUM[tag]["settings"][t]
CON = lambda tag, t, name: ST(tag, t)["contrasts"][name]
RES = {tag: P.MERGED if tag in dict(PAPER) else P.FAMILIES for tag, _ in MODELS}
BITS = {tag: load(RES[tag], tag)[1] for tag, _ in MODELS}
RECS = {tag: load(RES[tag], tag)[0] for tag, _ in MODELS}
DENSE = {tag: BITS[tag]["1.0|dense"]["test_q"] for tag, _ in MODELS}
DENSE_C = {tag: BITS[tag]["1.0|dense"]["test_c"] for tag, _ in MODELS}
PC = {}                                            # paper models: paired variants with cloze come from the cloze reruns
for tag, _ in PAPER:
    PC[tag] = load(P.PAIRED_CLOZE, f"{tag}_pc")[1]
for tag, _ in MODELS:
    assert SUM[tag]["known"] == int(DENSE[tag].sum()), tag      # the summary and the bits describe the same runs
STATS = json.loads((TAB / "stats.json").read_text())
EQ = json.loads((TAB / "entityquestions.json").read_text())
MIXR = json.loads((TAB / "mixture.json").read_text())

# ---------------------------------------------------------------- 1. dense models
num_range("SevKnown", [SUM[t]["known"] for t, _ in MODELS], nd=0)
num("SevKnownLoNum", f"{min(SUM[t]['known'] for t, _ in MODELS):,}")
num_range("SevParamKeepSeventy", [pct(ST(tag, "0.7")["param_keep"]) for tag, _ in MODELS], nd=0)
num_range("SevParamKeepFifty", [pct(ST(tag, "0.5")["param_keep"]) for tag, _ in MODELS], nd=0)

# ---------------------------------------------------------------- 2. generic repair restores perplexity, not facts
tz = [ST(tag, t)["teaser"] for tag, t in CELLS]
num("SevPplGapRec", f"{100 * np.mean([z['ppl_gap_recovered'] for z in tz]):.0f}\\%")
num("SevFactsRec", f"{100 * np.mean([z['facts_recovered'] for z in tz]):.0f}\\%")
num("SevFactsRecKG", f"{100 * np.mean([z['facts_recovered_kg'] for z in tz]):.0f}\\%")
# per model group: on the 1B models C4 repair recovers much more of the perplexity gap than of the lost facts; on the
# larger models it recovers little of either (the 14-setting mean alone would hide this)
tz_s = [ST(tag, t)["teaser"] for tag, t in CELLS if tag in SMALL]
tz_l = [ST(tag, t)["teaser"] for tag, t in CELLS if tag not in SMALL]
num_range("SmallPplRec", [100 * z["ppl_gap_recovered"] for z in tz_s], nd=0)
num_range("SmallFactsRec", [100 * z["facts_recovered"] for z in tz_s], nd=0, sign=True)
num("SmallFactsRecMax", max(100 * z["facts_recovered"] for z in tz_s), nd=0)
num_range("LargePplRec", [100 * z["ppl_gap_recovered"] for z in tz_l], nd=0, sign=True)
num_range("LargeFactsRec", [100 * z["facts_recovered"] for z in tz_l], nd=0, sign=True)
assert all(z["ppl_gap_recovered"] > z["facts_recovered"] + 0.1 for z in tz_s), "text: 1B ppl recovery well above facts"
assert all(abs(z["ppl_gap_recovered"]) < 0.2 and abs(z["facts_recovered"]) < 0.2 for z in tz_l), "text: larger models little of either"
cf4 = {c: CON(*c, "CfourRep") for c in CELLS}
cf4_neg = [c for c, d in cf4.items() if d["sig"] and d["diff"] < 0]
cf4_pos = [c for c, d in cf4.items() if d["sig"] and d["diff"] > 0]
num("SevCfourRepSigNeg", words(len(cf4_neg)))
num("SevCfourRepSigPos", words(len(cf4_pos)))
num_range("SevCfourRep", [pct(d["diff"]) for d in cf4.values()], sign=True)
num_range("NewRho", [ST(tag, t)["rho"] for tag, t in NCELLS], sign=True, nd=2)
rho_ex = [ST(tag, t)["rho"] for tag, t in CELLS if tag != "olmo7b"]
num("SevRhoMaxExOlmo", max(rho_ex), sign=True, nd=2)
mismatch = sum(ST(tag, t)["lowest_ppl"] != ST(tag, t)["best_ret"] for tag, t in CELLS)
num("SevMismatch", words(mismatch))

# ---------------------------------------------------------------- 3. channels on all seven models
chan = {name: {c: CON(*c, name) for c in CELLS} for name in ("CfourRep", "RepChan", "SelChan", "SelNoRep", "Both")}
num_range("SevRep", [pct(d["diff"]) for d in chan["RepChan"].values()])
num("SevRepSigPos", words(sum(d["sig"] and d["diff"] > 0 for d in chan["RepChan"].values())))
selsig = [c for c, d in chan["SelChan"].items() if d["sig"] and d["diff"] > 0]
num_range("SevSel", [pct(chan["SelChan"][c]["diff"]) for c in selsig])
num("SevSelSigPos", words(len(selsig)))
num_range("SevSelNoRep", [pct(d["diff"]) for d in chan["SelNoRep"].values()])
num_range("SevBoth", [pct(d["diff"]) for d in chan["Both"].values()])
num_range("NewBoth", [pct(chan["Both"][c]["diff"]) for c in NCELLS])
num_range("NewRep", [pct(chan["RepChan"][c]["diff"]) for c in NCELLS])
num_range("NewSel", [pct(chan["SelChan"][c]["diff"]) for c in NCELLS])
num("SevBothMean", float(np.mean([pct(d["diff"]) for d in chan["Both"].values()])), sign=True)
seed_runs = [(tag, t, s, ST(tag, t)["seeds"][s]["both_minus_c4"]) for tag, t in CELLS for s in ST(tag, t)["seeds"]]
num("SevSeedRuns", f"{len(seed_runs)}")
num_range("SevSeedBoth", [v for *_, v in seed_runs])
seeds_complete = all(len(ST(tag, t)["seeds"]) == 3 for tag, t in CELLS)
gm = [(tag, t, SUM[tag]["widths"]["gmle_both"][t]) for tag, t in CELLS
      if SUM[tag].get("widths") and SUM[tag]["widths"].get("gmle_both", {}).get(t)]
num("SevWidthN", words(len(gm)))
num("SevWidthSig", words(sum(d["sig"] and d["diff"] > 0 for *_, d in gm)))
num_range("SevWidthBoth", [pct(d["diff"]) for *_, d in gm])
# best configuration on the larger models
best_selnone = [c for c in LCELLS if ST(*c)["best_any"][1] == K["selnone"]]
num("LargeSelNoneBest", words(len(best_selnone)))
num("LargeN", words(len(LCELLS)))
# ridge strength and four times more C4 on the larger models
lam_ok = all(SUM[tag]["lam"]["1e-4"] > SUM[tag]["lam"]["1e-2"] > SUM[tag]["lam"]["1e-1"] for tag in LARGE)
c4x = {tag: SUM[tag]["cal4x"] for tag in LARGE}
cost = {tag: (c4x[tag]["ppl1"][1] - c4x[tag]["ppl1"][0], c4x[tag]["ppl4"][1] - c4x[tag]["ppl4"][0]) for tag in LARGE}
num_range("LargeCalFourRet", [pct(c4x[tag]["d"]["diff"]) for tag in LARGE], sign=True)
num("LargeCalFourSigNeg", words(sum(c4x[t]["d"]["sig"] and c4x[t]["d"]["diff"] < 0 for t in LARGE)))
num("LargeCalFourSigPos", words(sum(c4x[t]["d"]["sig"] and c4x[t]["d"]["diff"] > 0 for t in LARGE)))
num_range("LargeLamOneRet", [SUM[tag]["lam"]["1e-1"] - SUM[tag]["lam"]["1e-4"] for tag in LARGE], sign=True)
num_range("LargeLamDrop", [SUM[tag]["lam"]["1e-4"] - SUM[tag]["lam"]["1e-1"] for tag in LARGE])
# generic cost on the larger models (zero-shot)
zs = {c: ST(*c)["zs"] for c in LCELLS}
num_range("LargeHs", [z["hs"] for z in zs.values()], sign=True)
num_range("LargeHsDrop", [-z["hs"] for z in zs.values()])
num_range("LargeArc", [z["arc"] for z in zs.values()], sign=True)
num_range("LargeHsRep", [z["hs_rep"] for z in zs.values()], sign=True)
num_range("LargeRepChan", [pct(chan["RepChan"][c]["diff"]) for c in LCELLS])
# popularity quintiles at 70%
pop_up = pop_same = pop_down = 0
for tag, _ in MODELS:
    for n_, a_, b_ in ST(tag, "0.7")["pop"]:
        pop_up += round(b_, 6) > round(a_, 6)
        pop_same += round(b_, 6) == round(a_, 6)
        pop_down += round(b_, 6) < round(a_, 6)
num("SevPopUp", f"{pop_up}")
num("SevPopSame", words(pop_same))
num("SevPopCells", f"{5 * len(MODELS)}")

# ---------------------------------------------------------------- 4. Lemma 1 terms on the three further families
mech = {tag: SUM[tag]["mech"] for tag, _ in NEW}
num_range("NewMechCfour", [mech[t][b]["c4"] for t in mech for b in BUD], nd=2)
num_range("NewMechWeb", [mech[t][b]["c4web"] for t in mech for b in BUD], nd=2)
num_range("NewMechMix", [mech[t][b]["mix"] for t in mech for b in BUD], nd=2)
num_range("NewMechKGWeb", [mech[t][b]["kgweb"] for t in mech for b in BUD], nd=2)
num("NewMechTruthMax", f"{max(mech[t][b]['truth_absdiff'] for t in mech for b in BUD):.3f}")
worse = [mech[t][b]["c4_worse_blocks"] for t in mech for b in BUD]
num("NewMechWorse", f"{sum(w[0] for w in worse)}")
num("NewMechBlocks", f"{sum(w[1] for w in worse)}")
gap = max(SUM[t]["mech"]["identity_gap_max"] for t, _ in MODELS if SUM[t].get("mech"))
e = int(np.floor(np.log10(gap)))
num("SevGapMax", f"\\ensuremath{{{gap / 10 ** e:.1f}\\times 10^{{{e}}}}}")
truth_lower_all = all(all(mech[t][b]["truth_true_lower"]) for t in mech for b in BUD)
ngap = max(SUM[t]["mech"]["identity_gap_max"] for t, _ in NEW)
e = int(np.floor(np.log10(ngap)))
num("NewGapMax", f"\\ensuremath{{{ngap / 10 ** e:.1f}\\times 10^{{{e}}}}}")
num_range("NewWidthRatio", [SUM[t]["widths"]["ratio_mean"] for t, _ in NEW], nd=2)
num_range("NewWidthMoved", [100 * np.mean(list(SUM[t]["widths"]["moved"].values())) for t, _ in NEW])
num_range("NewWidthSeedMoved", [100 * SUM[t]["widths"]["seed_moved"] for t, _ in NEW])

# ---------------------------------------------------------------- 5. facts or formats on all seven models
def con_rng(name, cells, prefix, sign=True):
    ds = [CON(*c, name) for c in cells]
    num_range(prefix, [pct(d["diff"]) for d in ds], sign=sign)
    num(prefix + "SigPos", words(sum(d["sig"] and d["diff"] > 0 for d in ds)))
    num(prefix + "SigNeg", words(sum(d["sig"] and d["diff"] < 0 for d in ds)))
    num(prefix + "Pos", words(sum(d["diff"] > 0 for d in ds)))
    return ds


tdn = con_rng("TruthDecl", NCELLS, "NewTruthDecl")
tqn = con_rng("TruthQA", NCELLS, "NewTruthQA")
fmn = con_rng("Format", NCELLS, "NewFormat")
enn = con_rng("Ent", NCELLS, "NewEnt")
rln = con_rng("Rel", NCELLS, "NewRel")
wkn = con_rng("Wiki", NCELLS, "NewWiki")
wcn = con_rng("WikiClean", NCELLS, "NewWikiClean")
ann = con_rng("Anch", NCELLS, "NewAnch")
fms = con_rng("Format", [c for c in CELLS if c[0] in SMALL], "SmallFormat")
fml = con_rng("Format", LCELLS, "LargeFormat")
# truth over three calibration samples (two-level bootstrap, scripts/analysis/stats.py)
TP = STATS["truth"]["pooled"]
for grp, nm in (("new", "New"), ("all", "Sev")):
    v = TP[grp]
    num(f"{nm}TruthPoolMean", v["diff"], sign=True); num(f"{nm}TruthPoolLo", v["lo"], sign=True)
    num(f"{nm}TruthPoolHi", v["hi"], sign=True); num(f"{nm}TruthPoolLoNinety", v["lo90"], sign=True)
    num(f"{nm}TruthPoolHiNinety", v["hi90"], sign=True)
num("NewTruthRuns", f"{3 * 2 * len(NCELLS)}")
num("NewTruthEqBound", f"{TP['new']['eq_bound_smallest']:.1f}")
num("SevWikiHi", max(pct(CON(*c, "Wiki")["diff"]) for c in CELLS))
num("SevCovMax", max(abs(pct(CON(*c, n)["diff"])) for c in CELLS for n in ("Format", "Rel")))
large_fmt_pos = [c for c in LCELLS if CON(*c, "Format")["diff"] > 0]
num("SevTruthRuns", f"{3 * 2 * len(CELLS)}")
new_equiv = bool(TP["new"]["equivalent_at_margin"])
tcells = STATS["truth"]["cells"]
sig_true = [k for k, v in tcells.items() if v["sig"] and v["diff"] > 0]
sig_cf = [k for k, v in tcells.items() if v["sig"] and v["diff"] < 0]
num("SevTruthCellSigTrue", words(len(sig_true)))
num("SevTruthCellSigCf", words(len(sig_cf)))
num("SevTruthCells", f"{len(tcells)}")
# format by query type (scripts/analysis/stats.py)
FQ = STATS["format_query"]
fq_int = {k: v["interaction"] for k, v in FQ["cells"].items()}
num("FqIntPos", words(FQ["interaction_positive"]))
num("FqIntSig", words(sum(v["sig"] and v["diff"] > 0 for v in fq_int.values())))
num("FqN", words(len(fq_int)))
num("FqIntPool", FQ["pooled"]["all"]["interaction"]["diff"], sign=True)
num("FqIntPoolLo", FQ["pooled"]["all"]["interaction"]["lo"], sign=True)
num("FqIntPoolHi", FQ["pooled"]["all"]["interaction"]["hi"], sign=True)
num("FqMainSmall", FQ["pooled"]["small"]["main"]["diff"], sign=True)
num("FqMainLarge", FQ["pooled"]["large"]["main"]["diff"], sign=True)
num("FqQuestionLarge", FQ["pooled"]["large"]["question"]["diff"], sign=True)
num("FqClozeLarge", FQ["pooled"]["large"]["cloze"]["diff"], sign=True)
qa_helps = [k for k, v in FQ["cells"].items() if v["question"]["sig"] and v["question"]["diff"] > 0]
num("FqQaHelpsN", words(len(qa_helps)))
# Holm counts behind Table 3 (scripts/analysis/stats.py): every bold difference that does not survive Holm, per row
HOLM = STATS["holm"]
holm_drop = {}
for row in ("C4 repair vs none", "Repair channel", "Selection channel", "Selection, no repair", "Both channels"):
    drop = 0
    for g in ("paper", "new"):
        h = HOLM[f"question|{g}|{row}"]
        drop += (h["sig_pos"] - h["holm_pos"]) + (h["sig_neg"] - h["holm_neg"])
    holm_drop[row] = drop
num("HolmDropTableThree", words(sum(holm_drop.values())))


def holm_reject(ps, alpha=0.05):
    order, rej = np.argsort(ps), np.zeros(len(ps), bool)
    for r, i in enumerate(order):
        if ps[i] > alpha / (len(ps) - r):
            break
        rej[i] = True
    return rej


# Holm over the 14 settings of each row of Table 3, on the McNemar p-values of the paired comparisons
h_sig = h_keep = 0
for name in ("CfourRep", "RepChan", "SelChan", "SelNoRep", "Both"):
    ds = [CON(*c, name) for c in CELLS]
    rej = holm_reject(np.array([d["p"] for d in ds]))
    h_sig += sum(d["sig"] for d in ds)
    h_keep += sum(d["sig"] and r for d, r in zip(ds, rej))
num("HolmSigTableThree", f"{h_sig}")
num("HolmKeepTableThree", f"{h_keep}")

# ---------------------------------------------------------------- 6. tables with all seven models
heads = " & ".join(f"\\multicolumn{{2}}{{c}}{{{SHORT[tag]}}}" for tag, _ in MODELS)
rules = " ".join(f"\\cmidrule(lr){{{2 + 2 * k}-{3 + 2 * k}}}" for k in range(len(MODELS)))
budrow = " & ".join(f"{int(float(t) * 100)}" for _, t in CELLS)
HEAD = ["\\begin{tabular}{@{}l" + "c" * len(CELLS) + "@{}}", "\\toprule", " & " + heads + " \\\\", rules]
ROWS_T2 = [("none", "C4 / none"), ("c4", "C4 / C4"), ("rep", "C4 / KG"), ("sel", "KG / C4"), ("selnone", "KG / none"),
           ("both", "KG / KG"), ("qa", "KG-QA / KG-QA")]
t2 = HEAD + ["Units kept (\\%) & " + budrow + " \\\\", "\\midrule",
             "Dense-known facts & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{SUM[tag]['known']:,}}}" for tag, _ in MODELS) + " \\\\",
             "\\midrule"]
for c, name in ROWS_T2:
    t2.append(f"{name} & " + " & ".join(f"{ST(tag, t)['ret'][c]:.1f}" for tag, t in CELLS) + " \\\\")
(TAB / "table2_main7.tex").write_text("\n".join(t2 + ["\\bottomrule", "\\end{tabular}"]) + "\n")
T3 = [("CfourRep", "C4 repair vs none"), ("RepChan", "Repair channel"), ("SelChan", "Selection channel"),
      ("SelNoRep", "Selection, no repair"), ("Both", "Both channels")]
t3 = HEAD + ["Contrast \\hfill units kept (\\%) & " + budrow + " \\\\", "\\midrule"]
for name, lab in T3:
    cells = []
    for c in CELLS:
        d = chan[name][c]; v = f"{pct(d['diff']):+.1f}".replace("-", "$-$")
        cells.append(f"\\textbf{{{v}}}" if d["sig"] else v)
    t3.append(f"{lab} & " + " & ".join(cells) + " \\\\")
(TAB / "table3_contrasts7.tex").write_text("\n".join(t3 + ["\\bottomrule", "\\end{tabular}"]) + "\n")
tp = HEAD + ["Selection / repair \\hfill units kept (\\%) & " + budrow + " \\\\", "\\midrule",
             "Dense & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{SUM[tag]['dense_ppl']:.2f}}}" for tag, _ in MODELS) + " \\\\"]
for c, name in ROWS_T2:
    tp.append(f"{name} & " + " & ".join(f"{ST(tag, t)['ppl'][c]:.1f}" for tag, t in CELLS) + " \\\\")
(TAB / "tableA_perplexity7.tex").write_text("\n".join(tp + ["\\bottomrule", "\\end{tabular}"]) + "\n")


# cloze contrasts on all seven models (paired variants of the paper models from the cloze reruns)
def cbits(tag, t, c):
    key = f"{t}|{K[c]}"
    if tag in PC and c in ("dcf", "qacf", "syn"):
        return PC[tag][key]["test_c"]
    v = BITS[tag][key]["test_c"]
    return v if len(v) else None


CLOZE = [("CfourRep", "none", "c4", "C4 repair vs none"), ("RepChan", "c4", "rep", "Repair channel"),
         ("SelChan", "c4", "sel", "Selection channel"), ("SelNoRep", "none", "selnone", "Selection, no repair"),
         ("Both", "c4", "both", "Both channels"), ("TruthDecl", "dcf", "both", "Truth (declarative)"),
         ("TruthQA", "qacf", "qa", "Truth (question-answer)"), ("Format", "both", "qa", "Question-answer minus declarative"),
         ("Ent", "syn", "both", "Real minus made-up subjects"), ("Rel", "other", "both", "Tested minus other relations"),
         ("Wiki", "wiki", "both", "KG statements minus Wikipedia"), ("Anch", "both", "anch", "Anchored minus entity-disjoint")]
clo = {}
for name, a, b, _ in CLOZE:
    for tag, t in CELLS:
        xa, xb = cbits(tag, t, a), cbits(tag, t, b)
        if xa is not None and xb is not None:
            clo.setdefault(name, {})[(tag, t)] = paired(xa, xb, DENSE_C[tag])
tc = HEAD + ["Contrast (cloze queries) \\hfill units kept (\\%) & " + budrow + " \\\\", "\\midrule"]
for name, _, _, lab in CLOZE:
    cells = []
    for c in CELLS:
        if c not in clo.get(name, {}):
            cells.append(pending(f"cloze {name} {c}")); continue
        d = clo[name][c]; v = f"{pct(d['diff']):+.1f}".replace("-", "$-$")
        cells.append(f"\\textbf{{{v}}}" if d["sig"] else v)
    tc.append(f"{lab} & " + " & ".join(cells) + " \\\\")
(TAB / "tableA_cloze7.tex").write_text("\n".join(tc + ["\\bottomrule", "\\end{tabular}"]) + "\n")
num_range("NewCloBoth", [pct(clo["Both"][c]["diff"]) for c in NCELLS], sign=True)
num("NewCloBothSigPos", words(sum(clo["Both"][c]["sig"] and clo["Both"][c]["diff"] > 0 for c in NCELLS)))
num("NewCloFormatNeg", words(sum(clo["Format"][c]["diff"] < 0 for c in NCELLS)))
num("NewCloTruthSigCf", words(sum(clo[n][c]["sig"] and clo[n][c]["diff"] < 0 for n in ("TruthDecl", "TruthQA") for c in NCELLS)))
num("NewCloKnownLo", f"{min(int(DENSE_C[t].sum()) for t, _ in NEW):,}")
num("NewCloKnownHi", f"{max(int(DENSE_C[t].sum()) for t, _ in NEW):,}")

# popularity table (7 models) and the relation table of the three further families
pc_lines = ["\\begin{tabular}{@{}l" + "c" * len(MODELS) + "@{}}", "\\toprule",
            "Quintile & " + " & ".join(SHORT[t] for t, _ in MODELS) + " \\\\", "\\midrule"]
for q in range(5):
    pc_lines.append(f"{q + 1} & " + " & ".join(
        (lambda v: f"{v[1]:.0f} $\\to$ {v[2]:.0f} ({v[0]})")(ST(t, "0.7")["pop"][q]) for t, _ in MODELS) + " \\\\")
(TAB / "table_breakdown_popularity7.tex").write_text("\n".join(pc_lines + ["\\bottomrule", "\\end{tabular}"]) + "\n")
GUESS = {"occupation", "country", "sport", "color", "religion", "genre"}
from kgcal import data as KD                                       # noqa: E402
_rows = KD.load_popqa()
TEST = KD.facts_by_split(_rows, KD.make_split(_rows))["test"]
PROPS = np.array([r["prop"] for r in TEST])
rel_lines = ["\\begin{tabular}{@{}l" + "c" * len(NEW) + "@{}}", "\\toprule",
             "Relation & " + " & ".join(SHORT[t] for t, _ in NEW) + " \\\\", "\\midrule"]
for r_ in sorted(set(PROPS)):
    cells = []
    for tag, _ in NEW:
        m = DENSE[tag] & (PROPS == r_)
        if m.sum() < 10:
            cells.append("--"); continue
        a_, b_ = BITS[tag][f"0.7|{K['c4']}"]["test_q"][m].mean(), BITS[tag][f"0.7|{K['both']}"]["test_q"][m].mean()
        cells.append(f"{100 * a_:.0f} $\\to$ {100 * b_:.0f} ({int(m.sum())})")
    rel_lines.append(f"{r_}{'$^{*}$' if r_ in GUESS else ''} & " + " & ".join(cells) + " \\\\")
(TAB / "table_breakdown_relation_new.tex").write_text("\n".join(rel_lines + ["\\bottomrule", "\\end{tabular}"]) + "\n")

# the further families: budgets, ridge, four times more C4, Lemma 1 terms, widths
fam = ["\\begin{tabular}{@{}lccccccccc@{}}", "\\toprule",
       " & \\multicolumn{2}{c}{params kept (\\%)} & \\multicolumn{3}{c}{C4/C4 retention, ridge} & "
       "\\multicolumn{2}{c}{4$\\times$ C4} & \\multicolumn{2}{c}{units moved (\\%)} \\\\",
       "\\cmidrule(lr){2-3} \\cmidrule(lr){4-6} \\cmidrule(lr){7-8} \\cmidrule(lr){9-10}",
       "Model & 70\\% & 50\\% & $10^{-4}$ & $10^{-2}$ & $10^{-1}$ & repair vs none & ppl cost & GRIDE-MLE & redraw \\\\",
       "\\midrule"]
for tag, _ in MODELS:
    s_ = SUM[tag]
    lam = s_.get("lam", {})
    lamc = [f"{lam[k]:.1f}" if k in lam else "--" for k in ("1e-4", "1e-2", "1e-1")]
    if s_.get("cal4x"):
        d4 = s_["cal4x"]["d"]; v4 = f"{pct(d4['diff']):+.1f}".replace("-", "$-$")
        c4c = [f"\\textbf{{{v4}}}" if d4["sig"] else v4,
               f"{cost[tag][0]:+.2f} $\\to$ {cost[tag][1]:+.2f}".replace("-", "$-$") if tag in cost else
               f"{s_['cal4x']['ppl1'][1] - s_['cal4x']['ppl1'][0]:+.2f} $\\to$ {s_['cal4x']['ppl4'][1] - s_['cal4x']['ppl4'][0]:+.2f}".replace("-", "$-$")]
    else:
        c4c = ["--", "--"]
    wd = s_.get("widths") or {}
    wc = [f"{100 * np.mean(list(wd['moved'].values())):.1f}" if wd.get("moved") else "--",
          f"{100 * wd['seed_moved']:.1f}" if wd.get("seed_moved") is not None else "--"]
    fam.append(f"{LABEL[tag]} & {100 * ST(tag, '0.7')['param_keep']:.1f} & {100 * ST(tag, '0.5')['param_keep']:.1f} & "
               + " & ".join(lamc + c4c + wc) + " \\\\")
(TAB / "tableB_families.tex").write_text("\n".join(fam + ["\\bottomrule", "\\end{tabular}"]) + "\n")
mt = ["\\begin{tabular}{@{}lcccccccc@{}}", "\\toprule",
      " & \\multicolumn{2}{c}{C4 repair, knowledge} & \\multicolumn{2}{c}{C4 repair, Web} & "
      "\\multicolumn{2}{c}{MIX(KG-D) repair, knowledge} & \\multicolumn{2}{c}{KG-D repair, Web} \\\\",
      "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7} \\cmidrule(lr){8-9}",
      "Model & 70\\% & 50\\% & 70\\% & 50\\% & 70\\% & 50\\% & 70\\% & 50\\% \\\\", "\\midrule"]
for tag, _ in MODELS:
    mm = SUM[tag].get("mech")
    if not mm:
        continue
    mt.append(f"{LABEL[tag]} & " + " & ".join(f"{mm[b][k]:.2f}" for k in ("c4", "c4web", "mix", "kgweb") for b in BUD) + " \\\\")
(TAB / "tableB_mechanism7.tex").write_text("\n".join(mt + ["\\bottomrule", "\\end{tabular}"]) + "\n")

# ---------------------------------------------------------------- 7. Figure 4 with all fourteen settings
import matplotlib                                                  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
plt.rcParams.update({"font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False})
PROPERTY = [("TruthDecl", "Truth: true minus counterfactual (declarative)"),
            ("TruthQA", "Truth: true minus counterfactual (question-answer)"),
            ("Format", "Format: question-answer minus declarative"),
            ("Ent", "Entities: real minus made-up subjects"),
            ("Rel", "Relations: tested minus other relations"),
            ("Wiki", "Text: KG statements minus Wikipedia"),
            ("Anch", "Anchoring: anchored minus entity-disjoint")]
MK = {"qwen15b": "o", "olmo1b": "s", "qwen7b": "^", "olmo7b": "D", "llama8b": "v", "mistral7b": "P", "phi4": "X"}
COL = {"qwen15b": "#1f77b4", "olmo1b": "#d62728", "qwen7b": "#08519c", "olmo7b": "#a50f15",
       "llama8b": "#2ca02c", "mistral7b": "#9467bd", "phi4": "#ff7f0e"}
fig, ax = plt.subplots(figsize=(3.4, 3.75))
n = len(PROPERTY)
for ri, (name, lab) in enumerate(PROPERTY):
    y0 = n - 1 - ri
    for ci, (tag, t) in enumerate(CELLS):
        d = CON(tag, t, name)
        y = y0 + 0.38 - ci * (0.76 / (len(CELLS) - 1))
        alpha = 1.0 if t == "0.7" else 0.55
        ax.plot([pct(d["lo"]), pct(d["hi"])], [y, y], color=COL[tag], lw=0.7, alpha=alpha)
        ax.plot(pct(d["diff"]), y, marker=MK[tag], color=COL[tag], ms=2.6, ls="none", alpha=alpha,
                mfc=COL[tag] if t == "0.7" else "white")
ax.axvline(0, color="#777", lw=0.6)
ax.set_yticks(range(n)); ax.set_yticklabels([p[1].replace(": ", ":\n") for p in PROPERTY][::-1], fontsize=6.3)
ax.set_xlabel("Retention difference (points)", fontsize=7)
handles = [plt.Line2D([], [], marker=MK[t], color=COL[t], ls="none", ms=3.5, label=SHORT[t]) for t, _ in MODELS]
ax.legend(handles=handles, fontsize=6, frameon=False, loc="lower center", bbox_to_anchor=(0.30, 1.0), ncol=4,
          columnspacing=0.7, handletextpad=0.2)
fig.tight_layout()
fig.savefig(FIG / "fig4_facts_or_formats7.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(FIG / "fig4_facts_or_formats7.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
COLORS = {"C4": "#7f7f7f", "WIKI": "#8c564b", "KG-D": "#1f77b4", "KG-D~": "#ff7f0e", "KG-QA": "#2ca02c",
          "KG-QA~": "#98df8a", "KG-S": "#9467bd", "KG-Ac": "#d62728", "KG-O2": "#17becf"}
fig, axes = plt.subplots(1, len(NEW), figsize=(7.0, 2.1), squeeze=False)
for ax, (tag, label) in zip(axes[0], NEW):
    for k, r in RECS[tag].items():
        if not k.startswith("0.7|") or r.get("sel_rule") != "actnorm":
            continue
        src = (r.get("rep_src") or r.get("sel_src") or "C4").replace("MIX:", "")
        if src in ("KG-O", "KG-A"):
            continue
        rr = 100 * BITS[tag][k]["test_q"][DENSE[tag]].mean()
        ax.scatter(r["ppl_wikitext2"], rr, s=14, color=COLORS.get(src, "#333"), zorder=3,
                   marker="x" if r.get("rep") == "none" else "o")
        if r["cond"] == K["c4"]:
            ax.annotate("C4/C4", (r["ppl_wikitext2"], rr), xytext=(4, -9), textcoords="offset points", fontsize=6)
    ax.set_title(f"{label}, 70% FFN", fontsize=7)
    ax.set_xlabel("WikiText-2 perplexity", fontsize=6.5); ax.tick_params(labelsize=6)
axes[0][0].set_ylabel("Fact retention (%)", fontsize=7)
names = [("C4", "C4"), ("KG-D", "KG-D"), ("KG-D~", "KG-D$^{\\sim}$"), ("KG-QA", "KG-QA"), ("KG-QA~", "KG-QA$^{\\sim}$"),
         ("KG-S", "KG-S"), ("KG-O2", "KG-O"), ("KG-Ac", "KG-A"), ("WIKI", "WIKI")]
hs = [plt.Line2D([], [], marker="o", color=COLORS[k], ls="none", ms=4, label=lab) for k, lab in names]
hs += [plt.Line2D([], [], marker="x", color=COLORS["C4"], ls="none", ms=4, label="C4, no repair"),
       plt.Line2D([], [], marker="x", color=COLORS["KG-D"], ls="none", ms=4, label="KG-D, no repair")]
fig.legend(handles=hs, loc="lower center", ncol=len(hs), fontsize=6, frameon=False, columnspacing=0.9, handletextpad=0.2)
fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.savefig(FIG / "figA_ppl_vs_retention_new.pdf"); plt.close(fig)

exec(open(Path(__file__).resolve().parent / "make_numbers_experiments.py").read())    # section 8 (same namespace)

# ---------------------------------------------------------------- 9. wording guards (seven-model sentences)
GUARDS = [
    (mismatch == len(CELLS), "6.1 'in none of the 14 settings does the lowest-perplexity configuration keep the most facts'"),
    (all(d["diff"] > 0 and d["sig"] for d in chan["RepChan"].values()), "6.2 'repair channel ... in all 14 settings'"),
    (all(d["diff"] > 0 and d["sig"] for d in chan["Both"].values()), "6.2 'both channels ... in every setting'"),
    (set(c for c in CELLS if c not in selsig) == {("qwen7b", "0.7"), ("qwen7b", "0.5")},
     "6.2 'selection channel ... leaves Qwen2.5-7B unchanged'"),
    (not any(chan["SelChan"][("qwen7b", t)]["sig"] for t in BUD), "6.2 Qwen2.5-7B selection channel not significant"),
    (all(v > 0 for *_, v in seed_runs), "6.2 'positive in all runs over three calibration samples'"),
    (seeds_complete, "6.2 three calibration samples for every setting"),
    (all(d["sig"] and d["diff"] > 0 for *_, d in gm), "App. widths 'still beats C4/C4 in all settings'"),
    (lam_ok, "6.2 'a larger ridge penalty lowers retention further' on every larger model"),
    (all(cost[t][1] < cost[t][0] for t in LARGE), "6.2 'more C4 lowers the perplexity cost of the repair on all five larger models'"),
    (set(cf4_neg) == {("olmo7b", "0.7"), ("olmo7b", "0.5"), ("mistral7b", "0.7"), ("mistral7b", "0.5"), ("phi4", "0.7")},
     "6.1 'C4 repair lowers retention significantly on OLMo-2-7B and Mistral-7B at both budgets and Phi-4 at 70%'"),
    (pop_down == 0, "App. popularity 'raises ... and leaves it unchanged in the others' (no decrease)"),
    (truth_lower_all, "6.3 'true statements always slightly lower the error' on the further families"),
    (all(mech[t][b]["c4"] > 1 for t in mech for b in BUD), "6.3 'C4 repair leaves more than the discarded error' on the further families"),
    (sum(d["sig"] and d["diff"] < 0 for d in fmn) >= 4 and not any(d["diff"] > 0 for d in fmn),
     "6.4 'question-answer text keeps fewer facts than declarative text on all six settings of the further families'"),
    (sum(d["sig"] and d["diff"] > 0 for d in enn) >= 4, "6.4 'real subject names help on the further families'"),
    (all(d["sig"] and d["diff"] > 0 for d in rln), "6.4 'tested relations ... in all six settings of the further families'"),
    (all(d["sig"] and d["diff"] > 0 for d in wkn), "6.4 'KG statements beat Wikipedia in all six settings of the further families'"),
    (not any(d["diff"] > 0 for d in ann), "6.4 'anchoring never helps the further families'"),
    (FQ["rule_met_format_depends_on_query"], "6.4 pre-registered interaction rule met"),
    (FQ["pooled"]["small"]["main"]["diff"] > 0 > FQ["pooled"]["large"]["main"]["diff"], "6.4 'sign of the format effect depends on the model'"),
    (not sig_cf or len(sig_cf) <= 1, "App. statistics 'at most one setting significantly favours counterfactual text'"),
    (TP["new"]["lo"] > 0 and TP["new"]["eq_bound_smallest"] < 1.5, "6.4 new families: truth small but positive, bound below 1.5"),
    (len(best_selnone) * 2 == len(LCELLS), "7 'selection on the mixture without repair kept the most facts in half of their settings'"),
    (all(zs[c]["hs_rep"] > zs[c]["hs"] for c in LCELLS), "6.2 and 7 'the generic cost is smallest when the statements enter only the repair'"),
    (large_fmt_pos == [("olmo7b", "0.5")], "6.4 'question-answer text keeps more facts on the larger models only on OLMo-2-7B at 50%'"),
    (HOLM["question|paper|Anchoring"]["holm_pos"] == 0, "6.4 anchoring: 'none of which survives Holm correction'"),
    (all(CON(*c, "Wiki")["diff"] > 0 for c in CELLS) and all(CON(*c, "WikiClean")["diff"] > 0 for c in CELLS),
     "6.4 'KG statements keep at least as many facts as Wikipedia paragraphs in all 14 settings', also without co-occurrence facts"),
    (sum(CON(*c, "TruthDecl")["sig"] and CON(*c, "TruthDecl")["diff"] < 0 for c in CELLS) == 0 and
     sum(CON(*c, "TruthQA")["sig"] and CON(*c, "TruthQA")["diff"] < 0 for c in CELLS) == 0,
     "6.4 'none of these 28 differences significantly favours counterfactual text' (sample 0)"),
]
for ok, sentence in GUARDS:
    assert ok, f"text no longer matches the data, rewrite: {sentence}"

# ---------------------------------------------------------------- 10. write
with open(TAB / "numbers08.tex", "w") as f:
    f.write("% Generated by scripts/paper/make_numbers.py. Do not edit by hand.\n")
    for k in sorted(NUM):
        f.write(f"\\newcommand{{\\N{k}}}{{{NUM[k]}}}\n")
(TAB / "numbers08.json").write_text(json.dumps(NUM, indent=1))
(NOTES / "pending.md").write_text("# Pending cells and numbers (generated by scripts/paper/make_numbers.py)\n\n"
                                 + ("\n".join(f"- {p}" for p in PENDING) or "none") + "\n")
print(f"{len(NUM)} macros written; {len(PENDING)} pending items")
