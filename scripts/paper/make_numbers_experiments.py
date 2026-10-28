# Section 8 of make_numbers.py (executed in its namespace): EntityQuestions, mixture share, sequential statistics, OBS
# selection, the human check, the statistics tables, the validation checks and the controls for format, scoring and
# truth. Every number goes through num(); every missing run goes through pending().

def cell_txt(d, nd=1):
    v = f"{pct(d['diff']) if abs(d['diff']) <= 1 else d['diff']:+.{nd}f}".replace("-", "$-$")
    return f"\\textbf{{{v}}}" if d["sig"] else v


def pts_txt(d, nd=1):
    """d holds points already (stats.json and entityquestions.json store points)."""
    v = f"{d['diff']:+.{nd}f}".replace("-", "$-$")
    return f"\\textbf{{{v}}}" if d["sig"] else v


# ---------------------------------------------------------------- 8a. EntityQuestions
def eq_cells(part, grp, name):
    """Point differences per setting in CELLS order (None where the run is missing)."""
    out = {}
    for group, gm in (("paper", PAPER), ("new", NEW)):
        key = f"{part}|{grp}|{group}|{name}"
        if key not in EQ:
            continue
        tags = [t for t, _ in gm if t in EQ["dense_known"]]
        for i, (tag, t) in enumerate([(tag, t) for tag in tags for t in BUD]):
            out[(tag, t)] = EQ[key]["cells"][i]
    return out


EQK = EQ["dense_known"]
num_range("EqKnown", [EQK[t]["new"] for t, _ in MODELS if t in EQK], nd=0)
num("EqKnownLoNum", f"{min(EQK[t]['new'] for t, _ in MODELS if t in EQK):,}")
num("EqKnownHiNum", f"{max(EQK[t]['new'] for t, _ in MODELS if t in EQK):,}")
eqa = {n: eq_cells("a", "new relations", n) for n in ("C4 repair vs none", "Repair channel", "Selection, no repair (vs none)",
                                                    "Both channels", "Truth (declarative)")}
eqb = {n: eq_cells("b", "new relations", n) for n in ("Repair channel", "Selection, no repair (vs C4/C4)", "Both channels",
                                                    "Truth (declarative)", "Truth (question-answer)", "Format (QA - declarative)")}
eq_missing_a = [c for c in CELLS if c not in eqa["Both channels"]]
eq_missing_b = [c for c in CELLS if c not in eqb["Both channels"]]
for c in eq_missing_a:
    pending(f"EntityQuestions (a) {c}")
for c in eq_missing_b:
    pending(f"EntityQuestions (b) {c}")


def eq_counts(cells, prefix):
    vals = list(cells.values())
    num_range(prefix, [v["diff"] for v in vals], sign=True)
    num(prefix + "SigPos", words(sum(v["sig"] and v["diff"] > 0 for v in vals)))
    num(prefix + "SigNeg", words(sum(v["sig"] and v["diff"] < 0 for v in vals)))
    num(prefix + "N", words(len(vals)))
    return vals


eq_counts(eqa["Both channels"], "EqTrBoth")
eq_counts(eqa["Repair channel"], "EqTrRep")
eq_counts(eqa["C4 repair vs none"], "EqTrCfour")
eq_counts(eqa["Truth (declarative)"], "EqTrTruth")
eq_counts(eqb["Both channels"], "EqRpBoth")
eq_counts(eqb["Repair channel"], "EqRpRep")
eq_counts(eqb["Selection, no repair (vs C4/C4)"], "EqRpSel")
eq_counts({c: v for c, v in eqb["Format (QA - declarative)"].items() if c[0] in SMALL}, "EqRpFormatSmall")
eq_counts({c: v for c, v in eqb["Format (QA - declarative)"].items() if c[0] in LARGE}, "EqRpFormatLarge")
holm_eq = {}
for part, name in (("a", "Both channels"), ("b", "Both channels")):
    holm_eq[part] = sum(EQ[f"{part}|new relations|{g}|{name}"]["holm_pos"] for g in ("paper", "new")
                        if f"{part}|new relations|{g}|{name}" in EQ)
num("EqTrBothHolm", words(holm_eq["a"]))
num("EqRpBothHolm", words(holm_eq["b"]))
cf4_eq_neg = [c for c, v in eqa["C4 repair vs none"].items() if v["sig"] and v["diff"] < 0]
POOLEQ = EQ["truth_pooled"]
for grp, nm in (("paper", "Paper"), ("new", "New")):
    v = POOLEQ.get(f"{grp}|all")
    for k, f_ in (("Mean", "diff"), ("Lo", "lo"), ("Hi", "hi"), ("LoNinety", "lo90"), ("HiNinety", "hi90")):
        num(f"Eq{nm}TruthPool{k}", nfmt(v[f_], True, 2) if v else pending(f"EntityQuestions pooled truth, {grp} models"))
    num(f"Eq{nm}TruthEquiv", ("yes" if v["equivalent_within_margin"] else "no") if v else pending(f"EntityQuestions equivalence, {grp}"))
eq_paper_equiv = bool(POOLEQ.get("paper|all", {}).get("equivalent_within_margin"))
# tables: transfer and replication, 14 settings
def eq_table(rows, cells_of, fname):
    lines = HEAD + ["Contrast \\hfill units kept (\\%) & " + budrow + " \\\\", "\\midrule"]
    for name, lab in rows:
        cs = cells_of[name]
        lines.append(f"{lab} & " + " & ".join(pts_txt(cs[c]) if c in cs else pending(f"EntityQuestions {fname} {c}")
                                              for c in CELLS) + " \\\\")
    (TAB / fname).write_text("\n".join(lines + ["\\bottomrule", "\\end{tabular}"]) + "\n")


eq_table([("C4 repair vs none", "C4 repair vs none"), ("Repair channel", "Repair channel"),
          ("Selection, no repair (vs none)", "Selection, no repair"), ("Both channels", "Both channels"),
          ("Truth (declarative)", "Truth (declarative)")], eqa, "tableB_eq_transfer.tex")
eq_table([("Repair channel", "Repair channel"), ("Selection, no repair (vs C4/C4)", "KG / none vs C4 / C4"),
          ("Both channels", "Both channels"), ("Truth (declarative)", "Truth (declarative)"),
          ("Truth (question-answer)", "Truth (question-answer)"), ("Format (QA - declarative)", "Question-answer minus declarative")],
         eqb, "tableB_eq_replication.tex")
tt = ["\\begin{tabular}{@{}llcc@{}}", "\\toprule", "Model & Units kept & Declarative (samples 0, 1, 2) & Question-answer (samples 0, 1, 2) \\\\",
      "\\midrule"]
for tag, t in CELLS:
    v = EQ["truth_per_setting"].get(f"{tag}|{t}")
    if v is None:
        tt.append(f"{LABEL[tag]} & {int(float(t) * 100)}\\% & {pending(f'EntityQuestions truth samples {tag} {t}')} & {PEND} \\\\")
        continue
    f_ = lambda d: f"{pts_txt(d)} ({', '.join(f'{x:+.1f}' for x in d['samples'])})".replace("-", "$-$").replace("$$-$$", "$-$")
    tt.append(f"{LABEL[tag]} & {int(float(t) * 100)}\\% & {f_(v['decl'])} & {f_(v['qa'])} \\\\")
(TAB / "tableB_eq_truth.tex").write_text("\n".join(tt + ["\\bottomrule", "\\end{tabular}"]) + "\n")

# ---------------------------------------------------------------- 8b. mixture share (both channels, repair only, selection only)
SHARES = [(13, "Ten", "10"), (32, "TwentyFive", "25"), (64, "Fifty", "50"), (96, "SeventyFive", "75")]
MIXCH = {"both": ("R{n}|R{n}", K["both"]), "repair": ("R0|R{n}", K["rep"]), "selection": ("R{n}|none", K["selnone"])}


def mix_cell(tag, t, ch, n):
    mb = load(P.MIXTURE, f"{tag}_mix")
    if mb[1] is None:
        return None
    mr, mbits = mb
    dense = DENSE[tag]
    fmt, pkey = MIXCH[ch]
    key = f"{t}|{fmt.format(n=n)}"
    base_b, base_r = mbits[f"{t}|R0|R0"]["test_q"], mr[f"{t}|R0|R0"]
    if key in mbits:
        b, r = mbits[key]["test_q"], mr[key]
    elif n == 64:
        b, r = BITS[tag][f"{t}|{pkey}"]["test_q"], RECS[tag][f"{t}|{pkey}"]
    else:
        return None
    g = paired(base_b, b, dense)
    hs = None if r.get("hellaswag") is None or base_r.get("hellaswag") is None else 100 * (r["hellaswag"] - base_r["hellaswag"])
    return {"g": g, "hs": hs, "ppl": r["ppl_wikitext2"], "ppl0": base_r["ppl_wikitext2"]}


MIX = {(tag, t, ch, n): mix_cell(tag, t, ch, n) for tag, t in PCELLS for ch in MIXCH for n, *_ in SHARES}
for ch, nm in (("both", "Both"), ("repair", "Rep"), ("selection", "Sel")):
    for n, word, _ in SHARES:
        gs = [MIX[(tag, t, ch, n)]["g"]["diff"] for tag, t in PCELLS if MIX[(tag, t, ch, n)]]
        hs = [MIX[(tag, t, ch, n)]["hs"] for tag, t in PCELLS if tag in ("qwen7b", "olmo7b") and MIX[(tag, t, ch, n)]
              and MIX[(tag, t, ch, n)]["hs"] is not None]
        num(f"Mix{nm}Gain{word}", float(np.mean([pct(x) for x in gs])), sign=True)
        num(f"Mix{nm}Hs{word}", float(-np.mean(hs)) if len(hs) == 4 else pending(f"mixture HellaSwag {ch} {n}"))
mix_none_qualifies = all(MIXR[ch]["recommended_share"] is None for ch in MIXCH)
mix_gain_increasing = all(
    np.mean([MIX[(tag, t, "both", a)]["g"]["diff"] for tag, t in PCELLS]) < np.mean([MIX[(tag, t, "both", b)]["g"]["diff"] for tag, t in PCELLS])
    for (a, *_), (b, *_) in zip(SHARES, SHARES[1:]))
mix_hs_increasing = all(
    np.mean([-MIX[(tag, t, "both", a)]["hs"] for tag, t in PCELLS if tag in ("qwen7b", "olmo7b")])
    < np.mean([-MIX[(tag, t, "both", b)]["hs"] for tag, t in PCELLS if tag in ("qwen7b", "olmo7b")])
    for (a, *_), (b, *_) in zip(SHARES, SHARES[1:]))
for ch, nm in (("both", "Both"), ("repair", "Rep"), ("selection", "Sel")):
    v = MIXR[ch]["verdict"]["25%"]
    num(f"Mix{nm}RuleCount", words(v["settings_at_75pct"]))
ml = ["\\begin{tabular}{@{}llcccccccc@{}}", "\\toprule",
      " & & \\multicolumn{4}{c}{retention gain over C4/C4 (points)} & \\multicolumn{4}{c}{HellaSwag change (points)} \\\\",
      "\\cmidrule(lr){3-6} \\cmidrule(lr){7-10}",
      "Channel & Setting & 10\\% & 25\\% & 50\\% & 75\\% & 10\\% & 25\\% & 50\\% & 75\\% \\\\", "\\midrule"]
for ch, chlab in (("both", "both channels"), ("repair", "repair only"), ("selection", "selection only, no repair")):
    for tag, t in PCELLS:
        cs = [MIX[(tag, t, ch, n)] for n, *_ in SHARES]
        g = " & ".join(cell_txt(c["g"]) if c else PEND for c in cs)
        h = " & ".join((f"{c['hs']:+.1f}".replace("-", "$-$") if c and c["hs"] is not None else "--") for c in cs)
        ml.append(f"{chlab if (tag, t) == PCELLS[0] else ''} & {SHORT[tag]} {int(float(t) * 100)}\\% & {g} & {h} \\\\")
    means = [np.mean([pct(MIX[(tag, t, ch, n)]["g"]["diff"]) for tag, t in PCELLS]) for n, *_ in SHARES]
    hsm = [np.mean([MIX[(tag, t, ch, n)]["hs"] for tag, t in PCELLS if tag in ("qwen7b", "olmo7b")]) for n, *_ in SHARES]
    ml.append(" & mean & " + " & ".join(f"{m:+.1f}".replace("-", "$-$") for m in means) + " & "
              + " & ".join(f"{m:+.1f}".replace("-", "$-$") for m in hsm) + " \\\\")
    ml.append("\\midrule" if ch != "selection" else "\\bottomrule")
(TAB / "tableB_mixture.tex").write_text("\n".join(ml + ["\\end{tabular}"]) + "\n")

# ---------------------------------------------------------------- 8c. sequential statistics and 8d. OBS selection
def alt_run(sub, stem, tag):
    return load(P.RESULTS / sub, f"{tag}_{stem}")


SEQ, OBS = {}, {}
for tag, _ in PAPER:
    r_, b_ = alt_run("sequential", "seq", tag)
    for t in BUD:
        if b_ is not None and all(f"{t}|{k}" in b_ for k in (K["c4"], K["rep"], K["both"], K["dcf"])):
            SEQ[(tag, t)] = {"both": paired(b_[f"{t}|{K['c4']}"]["test_q"], b_[f"{t}|{K['both']}"]["test_q"], DENSE[tag]),
                             "rep": paired(b_[f"{t}|{K['c4']}"]["test_q"], b_[f"{t}|{K['rep']}"]["test_q"], DENSE[tag]),
                             "truth": paired(b_[f"{t}|{K['dcf']}"]["test_q"], b_[f"{t}|{K['both']}"]["test_q"], DENSE[tag]),
                             "c4": 100 * b_[f"{t}|{K['c4']}"]["test_q"][DENSE[tag]].mean()}
        else:
            pending(f"sequential pilot {tag} {t}")
    r_, b_ = alt_run("obs", "obs", tag)
    for t in BUD:
        if b_ is not None and all(f"{t}|{k}" in b_ for k in (K["c4"], K["rep"], K["both"], K["dcf"], K["selnone"])):
            OBS[(tag, t)] = {"both": paired(b_[f"{t}|{K['c4']}"]["test_q"], b_[f"{t}|{K['both']}"]["test_q"], DENSE[tag]),
                             "rep": paired(b_[f"{t}|{K['c4']}"]["test_q"], b_[f"{t}|{K['rep']}"]["test_q"], DENSE[tag]),
                             "selnone": paired(b_[f"{t}|{K['c4']}"]["test_q"], b_[f"{t}|{K['selnone']}"]["test_q"], DENSE[tag]),
                             "truth": paired(b_[f"{t}|{K['dcf']}"]["test_q"], b_[f"{t}|{K['both']}"]["test_q"], DENSE[tag]),
                             "c4": 100 * b_[f"{t}|{K['c4']}"]["test_q"][DENSE[tag]].mean(),
                             "hs": (lambda a, b: None if a is None or b is None else 100 * (b - a))(
                                 r_[f"{t}|{K['c4']}"].get("hellaswag"), r_[f"{t}|{K['both']}"].get("hellaswag"))}
        else:
            pending(f"OBS selection {tag} {t}")
for nm, D_ in (("Seq", SEQ), ("Obs", OBS)):
    num(f"{nm}N", words(len(D_)))
    num(f"{nm}BothSig", words(sum(v["both"]["sig"] and v["both"]["diff"] > 0 for v in D_.values())))
    num(f"{nm}RepSig", words(sum(v["rep"]["sig"] and v["rep"]["diff"] > 0 for v in D_.values())))
    num(f"{nm}RepPos", words(sum(v["rep"]["diff"] > 0 for v in D_.values())))
    num_range(f"{nm}Both", [pct(v["both"]["diff"]) for v in D_.values()], sign=True)
    num_range(f"{nm}Truth", [pct(v["truth"]["diff"]) for v in D_.values()], sign=True)
    num(f"{nm}TruthSigCf", words(sum(v["truth"]["sig"] and v["truth"]["diff"] < 0 for v in D_.values())))
    tags_done = sorted({tag for tag, _ in D_}, key=lambda x: [t for t, _ in PAPER].index(x))
    num(f"{nm}Models", "the four Qwen2.5 and OLMo-2 models" if len(tags_done) == 4 else
        ("the two 1B models" if tags_done == ["qwen15b", "olmo1b"] else ", ".join(LABEL[t] for t in tags_done)))
seq_lines = ["\\begin{tabular}{@{}lcccccccc@{}}", "\\toprule",
             " & \\multicolumn{2}{c}{C4/C4 retention (\\%)} & \\multicolumn{2}{c}{both channels} & \\multicolumn{2}{c}{repair channel} & "
             "\\multicolumn{2}{c}{truth (declarative)} \\\\",
             "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7} \\cmidrule(lr){8-9}",
             "Setting & dense & seq. & dense & seq. & dense & seq. & dense & seq. \\\\", "\\midrule"]
for tag, t in PCELLS:
    if (tag, t) not in SEQ:
        seq_lines.append(f"{SHORT[tag]} {int(float(t) * 100)}\\% & " + " & ".join([PEND] * 8) + " \\\\"); continue
    v = SEQ[(tag, t)]
    seq_lines.append(f"{SHORT[tag]} {int(float(t) * 100)}\\% & {ST(tag, t)['ret']['c4']:.1f} & {v['c4']:.1f} & "
                     f"{cell_txt(CON(tag, t, 'Both'))} & {cell_txt(v['both'])} & {cell_txt(CON(tag, t, 'RepChan'))} & {cell_txt(v['rep'])} & "
                     f"{cell_txt(CON(tag, t, 'TruthDecl'))} & {cell_txt(v['truth'])} \\\\")
(TAB / "tableB_sequential.tex").write_text("\n".join(seq_lines + ["\\bottomrule", "\\end{tabular}"]) + "\n")
obs_lines = ["\\begin{tabular}{@{}lcccccc@{}}", "\\toprule",
             "Setting & C4/C4 (\\%) & repair channel & KG / none & both channels & truth (decl.) & HellaSwag, both \\\\", "\\midrule"]
for tag, t in PCELLS:
    if (tag, t) not in OBS:
        obs_lines.append(f"{SHORT[tag]} {int(float(t) * 100)}\\% & " + " & ".join([PEND] * 6) + " \\\\"); continue
    v = OBS[(tag, t)]
    obs_lines.append(f"{SHORT[tag]} {int(float(t) * 100)}\\% & {v['c4']:.1f} & {cell_txt(v['rep'])} & {cell_txt(v['selnone'])} & "
                     f"{cell_txt(v['both'])} & {cell_txt(v['truth'])} & "
                     + (f"{v['hs']:+.1f}".replace("-", "$-$") if v["hs"] is not None else "--") + " \\\\")
(TAB / "tableB_obs.tex").write_text("\n".join(obs_lines + ["\\bottomrule", "\\end{tabular}"]) + "\n")

# ---------------------------------------------------------------- 8e. human check of the scoring rules
hk = json.loads((P.ANSWERS / "human_check_key.json").read_text())
hl = {int(r["id"]): r["correct"] == "y" for r in csv.DictReader(open(P.ANSWERS / "human_labels.csv"))}
assert len(hl) == len(hk["items"]) == 120
RULES = [("alias", "Alias"), ("word", "Word"), ("exact", "Exact"), ("judge", "Judge"), ("judge_llama", "Llama")]
by = {}
for i, v in hk["items"].items():
    by.setdefault(v["stratum"], []).append((hl[int(i)], v["rules"]))
NTOT = sum(hk["pool_sizes"].values())
acc = {}
for r, nm in RULES:
    acc[r] = sum(hk["pool_sizes"][s] / NTOT * np.mean([h == rr[r] for h, rr in xs]) for s, xs in by.items())
    num(f"Human{nm}", f"{100 * acc[r]:.0f}")
num("HumanN", f"{len(hl)}")
a_fp = sum(hk["pool_sizes"][s_] * np.mean([(not h) and rr["alias"] for h, rr in xs]) for s_, xs in by.items())
a_fn = sum(hk["pool_sizes"][s_] * np.mean([h and (not rr["alias"]) for h, rr in xs]) for s_, xs in by.items())
num("HumanAliasFalse", f"{a_fp:.0f}")
num("HumanAliasMiss", f"{a_fn:.0f}")
num("HumanPool", f"{NTOT:,}")
hum = ["\\begin{tabular}{@{}lrrr" + "r" * len(RULES) + "@{}}", "\\toprule",
       "Stratum & answers & sampled & human: correct & " + " & ".join(nm.replace("Judge", "Phi-4 judge").replace("Llama", "Llama judge")
                                                                    for _, nm in RULES) + " \\\\", "\\midrule"]
SNAME = {"agree": "all rules agree", "alias_only": "only alias accepts", "judge_only": "only the judges accept",
         "strict_only": "rules disagree otherwise"}
for s, xs in by.items():
    hum.append(f"{SNAME.get(s, s)} & {hk['pool_sizes'][s]:,} & {len(xs)} & {sum(h for h, _ in xs)} & "
               + " & ".join(f"{sum(h == rr[r] for h, rr in xs)}" for r, _ in RULES) + " \\\\")
hum.append("\\midrule")
hum.append(f"estimated agreement (\\%) & {NTOT:,} & {len(hl)} & & " + " & ".join(f"{100 * acc[r]:.0f}" for r, _ in RULES) + " \\\\")
(TAB / "tableB_human.tex").write_text("\n".join(hum + ["\\bottomrule", "\\end{tabular}"]) + "\n")
human_order = sorted(acc, key=lambda r: -acc[r])

# ---------------------------------------------------------------- 8f. statistics tables: truth per setting, Holm, format by query
ts = ["\\begin{tabular}{@{}llcc@{}}", "\\toprule", "Model & Units kept & Declarative & Question-answer \\\\", "\\midrule"]
for tag, t in CELLS:
    d, q = tcells[f"{tag}|{t}|decl"], tcells[f"{tag}|{t}|qa"]
    f_ = lambda v: (f"{pts_txt(v)} [{v['lo']:+.1f}, {v['hi']:+.1f}]").replace("-", "$-$").replace("$$-$$", "$-$")
    ts.append(f"{LABEL[tag]} & {int(float(t) * 100)}\\% & {f_(d)} & {f_(q)} \\\\")
(TAB / "tableB_truth_setting.tex").write_text("\n".join(ts + ["\\bottomrule", "\\end{tabular}"]) + "\n")
FAMS = ["C4 repair vs none", "Repair channel", "Selection channel", "Selection, no repair", "Both channels",
        "Truth (declarative), sample 0", "Truth (question-answer), sample 0", "Format (QA - declarative)",
        "Entities (real - made-up)", "Relations (tested - other)", "KG statements - Wikipedia", "Anchoring"]
hl_ = ["\\begin{tabular}{@{}lcccc@{}}", "\\toprule",
       " & \\multicolumn{2}{c}{question queries} & \\multicolumn{2}{c}{cloze queries} \\\\",
       "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}", "Contrast & Qwen2.5, OLMo-2 & further families & Qwen2.5, OLMo-2 & further families \\\\",
       "\\midrule"]
for fname in FAMS:
    cells = []
    for q in ("question", "cloze"):
        for g in ("paper", "new"):
            h = HOLM.get(f"{q}|{g}|{fname}")
            cells.append("--" if h is None else f"{h['sig_pos']}/{h['sig_neg']} ({h['holm_pos']}/{h['holm_neg']})")
    hl_.append(f"{fname.replace(' - ', ' minus ').replace(', sample 0', '')} & " + " & ".join(cells) + " \\\\")
(TAB / "tableB_holm.tex").write_text("\n".join(hl_ + ["\\bottomrule", "\\end{tabular}"]) + "\n")
fq = ["\\begin{tabular}{@{}llcccc@{}}", "\\toprule",
      "Model & Units kept & question queries & cloze queries & mean & question minus cloze \\\\", "\\midrule"]
for tag, t in CELLS:
    v = FQ["cells"].get(f"{tag}|{float(t)}") or FQ["cells"].get(f"{tag}|{t}")
    fq.append(f"{LABEL[tag]} & {int(float(t) * 100)}\\% & " + " & ".join(pts_txt(v[k]) for k in ("question", "cloze", "main", "interaction")) + " \\\\")
fq.append("\\midrule")
for g, lab in (("small", "1B models"), ("large", "7B to 14B models"), ("all", "all settings")):
    v = FQ["pooled"][g]
    fq.append(f"\\multicolumn{{2}}{{@{{}}l}}{{pooled, {lab}}} & " + " & ".join(pts_txt(v[k]) for k in ("question", "cloze", "main", "interaction")) + " \\\\")
(TAB / "tableB_format_query.tex").write_text("\n".join(fq + ["\\bottomrule", "\\end{tabular}"]) + "\n")

# ---------------------------------------------------------------- 8g. SparseGPT, inference cost, scoring rules, moments
# These cover Qwen2.5 and OLMo-2 only; the text reports them with the macros of numbers.tex (paper_models.py).

tr_sig = sum(HOLM[f"question|{g}|Truth ({f}), sample 0"]["sig_pos"] + HOLM[f"question|{g}|Truth ({f}), sample 0"]["sig_neg"]
             for g in ("paper", "new") for f in ("declarative", "question-answer"))
tr_holm = sum(HOLM[f"question|{g}|Truth ({f}), sample 0"]["holm_pos"] + HOLM[f"question|{g}|Truth ({f}), sample 0"]["holm_neg"]
              for g in ("paper", "new") for f in ("declarative", "question-answer"))
num("HolmTruthSig", words(tr_sig))
num("HolmTruthKeep", words(tr_holm))
num("HolmAnchSig", words(HOLM["question|paper|Anchoring"]["sig_pos"]))
chan_rows = ("Repair channel", "Selection channel", "Selection, no repair", "Both channels")
ch_sig = sum(HOLM[f"question|{g}|{r}"]["sig_pos"] for g in ("paper", "new") for r in chan_rows)
ch_keep = sum(HOLM[f"question|{g}|{r}"]["holm_pos"] for g in ("paper", "new") for r in chan_rows)
num("HolmChanSig", f"{ch_sig}")
num("HolmChanKeep", f"{ch_keep}")

# ---------------------------------------------------------------- 8i. overlap and coverage (overlap_coverage.py)
RV = json.loads((TAB / "overlap.json").read_text())
ov = RV["overlap"]
num("EqOvQ", f"{ov['questions']:,}"); num("EqOvUniq", f"{ov['unique_subjects']:,}")
num("EqOvTextQ", f"{ov['subject_in_text']['questions']}"); num("EqOvTextU", f"{ov['subject_in_text']['unique']}")
num("EqOvEntQ", f"{ov['subject_is_calibration_entity']['questions']}"); num("EqOvEntU", f"{ov['subject_is_calibration_entity']['unique']}")
num("EqOvEitherQ", f"{ov['subject_either']['questions']}"); num("EqOvEitherU", f"{ov['subject_either']['unique']}")
num("EqSubjDisjN", f"{ov['subject_disjoint_questions']:,}")
num("EqAnsQ", f"{ov['answer_in_text']['questions']}"); num("EqAnsDisjN", f"{ov['answer_disjoint_questions']:,}")
num("EqBothDisjN", f"{ov['both_disjoint_questions']:,}")
num("EqOvEntAsAnswer", f"{ov['subject_is_calibration_entity']['as_answer_alias']}")
num("EqOvEntAsSubject", words(ov['subject_is_calibration_entity']['as_subject_name']))
sub = {nm: [RV["eq_subsets"][f"{tag}|{t}|{nm}"] for tag, t in CELLS] for nm in ("subject_disjoint", "answer_disjoint", "both_disjoint")}
for nm, pre in (("subject_disjoint", "EqSubjDisj"), ("answer_disjoint", "EqAnsDisj"), ("both_disjoint", "EqBothDisj")):
    num_range(pre + "Both", [v["diff"] for v in sub[nm]], sign=True)
    num(pre + "Sig", words(sum(v["sig"] and v["diff"] > 0 for v in sub[nm])))
ans_ns = [c for c, v in zip(CELLS, sub["answer_disjoint"]) if not v["sig"]]
num("EqAnsDisjNotSig", " and ".join(f"{LABEL[c[0]]} at {int(float(c[1]) * 100)}\\%" for c in ans_ns) or "none")
FC = RV["format_coverage"]
num("FmtFactsDecl", f"{FC['facts_decl']:,}"); num("FmtFactsQA", f"{FC['facts_qa']:,}")
num("FmtRelTV", f"{FC['relation_tv']:.2f}")
bd = FC["bound"]
small_b = [bd[f"{t}|{b_}"]["coverage_bound"] for t in SMALL for b_ in BUD]
large_b = [bd[f"{t}|{b_}"]["coverage_bound"] for t in ("qwen7b", "olmo7b") for b_ in BUD]
num_range("FmtCovSmall", small_b, sign=True)
num_range("FmtCovLarge", large_b, sign=True)
num_range("FmtObsLarge", [bd[f"{t}|{b_}"]["format_contrast"] for t in ("qwen7b", "olmo7b") for b_ in BUD], sign=True)
# format itself matters where question-answer text wins by more than the coverage bound; the contrast is within reach of
# coverage alone where it lies between minus the bound and zero
own = [k for k, v in bd.items() if v["format_contrast"] > max(v["coverage_bound"], 0)]
cov_only = [k for k, v in bd.items() if -v["coverage_bound"] <= v["format_contrast"] <= 0.05]
num("FmtOwnN", words(len(own))); num("FmtCovOnlyN", words(len(cov_only)))

# ---------------------------------------------------------------- 8j. inference and theory checks (scripts/analysis/validation.py)
import math                                                       # noqa: E402
VA = json.loads((TAB / "validation.json").read_text())
CL, LE = VA["clustered"], VA["local_error"]
mi = lambda v, nd=1: f"{v:+.{nd}f}".replace("-", "$-$")
ci = lambda lo, hi, nd=1: f"[{lo:.{nd}f}, {hi:.{nd}f}]".replace("-", "$-$")
num_range("ClustWider", [100 * (v["width_ratio"] - 1) for v in CL["summary"].values()], nd=0)
num("ClustFacts", f"{CL['facts']:,}"); num("ClustSubjects", f"{CL['subjects']:,}")
for name, pre in (("Repair channel", "Rep"), ("Selection channel", "Sel"), ("Both channels", "Both")):
    num(f"Clust{pre}Sig", f"{CL['summary'][name]['sig_subj_pos']}")
for g, pre in (("paper", "Paper"), ("new", "New"), ("all", "All")):
    for k, suf in (("lo", "Lo"), ("hi", "Hi"), ("lo90", "LoNinety"), ("hi90", "HiNinety")):
        num(f"ClustTruth{pre}{suf}", CL["pooled_truth"][g][k], sign=True, nd=2)
PCN = CL["pooled_contrast"]
for name, pre in (("Both channels", "Both"), ("Repair channel", "Rep"), ("Selection channel", "Sel")):
    num(f"Pool{pre}", PCN[name]["mean"], sign=True)
    num(f"Pool{pre}Lo", PCN[name]["lo"], sign=True); num(f"Pool{pre}Hi", PCN[name]["hi"], sign=True)
ts_both = CL["truth_setting_both_formats"]
num("ClustTruthSetSig", words(sum(v["lo"] > 0 for v in ts_both.values())))
num("LocErrConc", f"{LE['concordant']}"); num("LocErrPairs", f"{LE['pairs']}")
num("LocErrShare", f"{round(100 * LE['share'])}")
num("LocErrP", f"10^{{{math.floor(math.log10(LE['sign_test_p']))}}}")
num("LocErrModels", words(len(LE["per_model"])))
num("LocErrOlmoSv", f"{words(LE['per_model']['olmo7b']['concordant'])} of {words(LE['per_model']['olmo7b']['pairs'])}")
# main-text Table 3: one row per contrast over the 14 settings, intervals that resample test subjects
T3 = [("C4 repair vs none", "C4 repair vs none", "CfourRep"), ("Repair channel", "Repair channel", "RepChan"),
      ("Selection channel", "Selection channel", "SelChan"), ("Selection, no repair", "Selection, no repair", "SelNoRep"),
      ("Both channels", "Both channels", "Both"), ("Truth, declarative", "Truth (declarative)", None),
      ("Truth, question-answer", "Truth (question-answer)", None)]
t3 = ["\\begin{tabular}{@{}lccc@{}}", "\\toprule", "Contrast & Range & $+$/$-$ & Mean [95\\% CI] \\\\", "\\midrule"]
t3_ranges = {}
for lab, name, con in T3:
    if con is None:
        cs = [CL["truth_cell_three_samples"][f"{tag}|{t}|{name}"] for tag, t in CELLS]
        lo_k, hi_k, pc, nd = "lo", "hi", PCN[name + ", three samples"], 2
    else:
        cs = [CL["cells"][f"{tag}|{t}|{name}"] for tag, t in CELLS]
        lo_k, hi_k, pc, nd = "subj_lo", "subj_hi", PCN[name], 1
    ds = [c["diff"] for c in cs]
    t3_ranges[lab] = (min(ds), max(ds), sum(c[lo_k] > 0 for c in cs), sum(c[hi_k] < 0 for c in cs))
    t3.append(f"{lab} & {mi(min(ds))} to {mi(max(ds))} & {t3_ranges[lab][2]}/{t3_ranges[lab][3]} & "
              f"{mi(pc['mean'], nd)} {ci(pc['lo'], pc['hi'], nd)} \\\\")
    if lab == "Both channels":
        t3.append("\\midrule")
(TAB / "table3_summary.tex").write_text("\n".join(t3 + ["\\bottomrule", "\\end{tabular}"]) + "\n")
for lab, pre in (("Truth, declarative", "TabTruthDecl"), ("Truth, question-answer", "TabTruthQA")):
    num(pre + "Lo", t3_ranges[lab][0], sign=True); num(pre + "Hi", t3_ranges[lab][1], sign=True)
    num(pre + "Pos", words(t3_ranges[lab][2]))
# appendix table: per setting, both channels and truth with subject-clustered intervals, and the error-retention pairs
ta = ["\\begin{tabular}{@{}llccc@{}}", "\\toprule", "Model & Kept & Both channels & Truth & Error \\\\", "\\midrule"]
for tag, t in CELLS:
    b_, tr = CL["cells"][f"{tag}|{t}|Both channels"], CL["truth_setting_both_formats"][f"{tag}|{t}"]
    le = LE["per_setting"].get(f"{tag}|{t}")
    ta.append(f"{SHORT[tag]} & {int(float(t) * 100)}\\% & {mi(b_['diff'])} {ci(b_['subj_lo'], b_['subj_hi'])} & "
              f"{mi(tr['diff'])} {ci(tr['lo'], tr['hi'])} & " + (f"{le['concordant']}/{le['pairs']}" if le else "--") + " \\\\")
ta.append("\\midrule")
for g, lab, tags in (("paper", "Qwen2.5, OLMo-2", [m for m, _ in PAPER]), ("new", "further families", [m for m, _ in NEW]),
                     ("all", "all seven models", [m for m, _ in MODELS])):
    b_, tr = CL["pooled_group"][f"{g}|Both channels"], CL["pooled_truth"][g]
    c_ = sum(LE["per_model"][m]["concordant"] for m in tags if m in LE["per_model"])
    n_ = sum(LE["per_model"][m]["pairs"] for m in tags if m in LE["per_model"])
    ta.append(f"\\multicolumn{{2}}{{@{{}}l}}{{Mean, {lab}}} & {mi(b_['mean'])} {ci(b_['lo'], b_['hi'])} & "
              f"{mi(tr['mean'], 2)} {ci(tr['lo'], tr['hi'], 2)} & {c_}/{n_} \\\\")
(TAB / "tableA_validation.tex").write_text("\n".join(ta + ["\\bottomrule", "\\end{tabular}"]) + "\n")
le_share = {m: v["concordant"] / v["pairs"] for m, v in LE["per_model"].items()}
GUARDS_8J = [
    (CL["summary"]["Both channels"]["sig_subj_pos"] == 14 and CL["summary"]["Repair channel"]["sig_subj_pos"] == 14,
     "6.4/App. A 'both channels and the repair channel stay significant in all 14 settings with subject clustering'"),
    (CL["summary"]["Selection channel"]["sig_subj"] == CL["summary"]["Selection channel"]["sig_fact"],
     "App. A 'the selection channel keeps the same count'"),
    (all(v["lo"] > 0 and v["hi90"] < 1.5 for v in CL["pooled_truth"].values()),
     "6.4/App. A 'clustered pooled truth effects stay positive and their 90% intervals stay below 1.5'"),
    (not any(v["hi"] < 0 for v in ts_both.values()), "App. A 'the truth difference is significantly negative in no setting'"),
    (LE["share"] > 0.75 and min(le_share, key=le_share.get) == "olmo7b", "6.3 'lower local error goes with more facts; OLMo-2-7B is the exception'"),
    (all(t3_ranges[lab][3] == 0 for lab in ("Repair channel", "Selection channel", "Selection, no repair", "Both channels",
                                            "Truth, declarative", "Truth, question-answer")),
     "Table 3 'no channel or truth difference is significantly negative'"),
    (all(abs(round(t3_ranges[lab][0], 1) - round(100 * min(CON(*c, con)["diff"] for c in CELLS), 1)) < 1e-9 and
         abs(round(t3_ranges[lab][1], 1) - round(100 * max(CON(*c, con)["diff"] for c in CELLS), 1)) < 1e-9
         for lab, _, con in T3 if con), "Table 3 ranges equal the ranges of the text macros"),
]
for ok, sentence in GUARDS_8J:
    assert ok, f"text no longer matches the data, rewrite: {sentence}"

# ---------------------------------------------------------------- 8k. controls for format, scoring and truth (controls.py)
RX = json.loads((TAB / "controls.json").read_text())
for item in RX["missing"]:
    pending(f"control run {item}")
GN = {"small": "Small", "large": "Large", "all": "All", "new": "New", "seven": "Seven"}


def rx_num(name, res, g, nd=1, ninety=False):
    """Pooled mean, interval and significance counts of one group, or placeholders while runs are missing."""
    v = res.get(g)
    keys = ("", "Lo", "Hi", "Pos", "Neg", "Min", "Max") + (("LoNinety", "HiNinety", "Equiv") if ninety else ())
    if v is None:
        for k in keys:
            num(name + k, PEND)
        return None
    num(name, v["diff"], sign=True, nd=nd); num(name + "Lo", v["lo"], sign=True, nd=nd); num(name + "Hi", v["hi"], sign=True, nd=nd)
    num(name + "Pos", words(v["sig_pos"])); num(name + "Neg", words(v["sig_neg"]))
    num(name + "Min", v["min"], sign=True, nd=nd); num(name + "Max", v["max"], sign=True, nd=nd)
    if ninety:
        num(name + "LoNinety", v["lo90"], sign=True, nd=nd); num(name + "HiNinety", v["hi90"], sign=True, nd=nd)
        num(name + "Equiv", "lies" if v["equiv"] else "does not lie")
    return v


rx_cell = lambda res, g: PEND if g not in res else f"{mi(res[g]['diff'])} ({res[g]['sig_pos']}/{res[g]['sig_neg']})"
FX, MX, TX = RX["fmt"], RX["mc"], RX["truth"]
inf = lambda tag, exp: RX["info"].get(f"{tag}|{exp}")
for tag, pre in (("qwen15b", "Qwen"), ("olmo1b", "Olmo")):
    i_ = inf(tag, "fmt")
    num(f"FmtSFacts{pre}", f"{i_['facts']:,}" if i_ else PEND)
i_ = inf("qwen15b", "fmt")
for k, nm in (("QA_S", "QA"), ("D_S", "D"), ("D_Srep", "Rep")):
    num(f"FmtSTok{nm}", f"{i_['tokens'][k]:,}" if i_ else PEND)
fall = [inf(t, "fmt") for t in ("qwen15b", "olmo1b", "qwen7b", "olmo7b", "llama8b", "mistral7b", "phi4")]
if all(fall):
    num_range("FmtSFacts", [x["facts"] for x in fall], nd=0)
else:
    num("FmtSFactsLo", PEND); num("FmtSFactsHi", PEND)
fi_ = [inf(t, "fmt") for t in ("qwen15b", "olmo1b")]
num("FmtSFactsAbout", f"{int(round(np.mean([x['facts'] for x in fi_]), -1)):,}" if all(fi_) else PEND)
num("FmtSTwice", f"{i_['D_Srep_facts_twice']:,}" if i_ else PEND)
for g in ("small", "large", "all", "new", "seven"):
    rx_num(f"FmtPrim{GN[g]}", FX["primary"], g)
    rx_num(f"FmtQaD{GN[g]}", FX["qa_vs_d"], g)
    rx_num(f"FmtCloze{GN[g]}", FX["primary_cloze"], g)
    rx_num(f"FmtInter{GN[g]}", FX["interaction"], g)
    rx_num(f"FmtMore{GN[g]}", FX["more_facts"], g)
for g in ("small", "large", "new"):
    r_ = FX["rule"].get(g)
    num(f"FmtRule{GN[g]}", PEND if r_ is None else ("met" if r_["format_effect"] else "not met"))
    num(f"FmtRuleDir{GN[g]}", PEND if r_ is None else r_["direction"])
dn = MX["dense"]
if len(dn) < 7:
    pending("multiple-choice dense accuracy range, all seven models")
if dn:
    num_range("McDense", [100 * v["mc_acc"] for v in dn.values()], nd=0)
    num_range("McGenAlso", [100 * v["gen_known_also_mc"] for v in dn.values()], nd=0)
else:
    for k in ("McDenseLo", "McDenseHi", "McGenAlsoLo", "McGenAlsoHi"):
        num(k, PEND)
for name, pre in (("C4 repair vs none", "McCfourRep"), ("Repair channel", "McRep"), ("Selection channel", "McSel"),
                  ("Selection, no repair", "McSelNone"), ("Both channels", "McBoth"), ("Truth (declarative)", "McTruthDecl"),
                  ("Truth (question-answer)", "McTruthQA"), ("Format (QA - declarative)", "McFormat")):
    for g in ("small", "large", "all", "new", "seven"):
        rx_num(f"{pre}{GN[g]}", MX[name], g)
for g, suf in (("all", ""), ("new", "New"), ("seven", "Seven")):     # McTruthPool: Qwen2.5 and OLMo-2
    tp = MX["truth_pooled"].get(g)
    rx_num("McTruthPool" + suf, {g: dict(tp, sig_pos=0, sig_neg=0, min=0, max=0)} if tp else {}, g, nd=2, ninety=True)
for g, suf in (("all", ""), ("new", "New")):
    r_ = MX["rule"].get(g)
    num("McRule" + suf, PEND if r_ is None else ("met" if r_["findings_survive"] else "not met"))
VLAB = {"V25": "TrQuarter", "V50": "TrHalf", "V75": "TrThreeQ", "V100": "TrAll", "VX": "TrImpl", "VC": "TrContra"}
for v_, pre in VLAB.items():
    for g in ("small", "large", "all", "new", "seven"):
        rx_num(f"{pre}{GN[g]}", TX[v_], g, nd=2, ninety=True)
    rx_num(f"{pre}Cloze", TX[v_ + " cloze"], "all", nd=2, ninety=True)
for g in ("small", "large", "all", "new", "seven"):
    rx_num(f"TrSlope{GN[g]}", TX["slope"], g, nd=2)
num("TrImplNewAbs", f"{TX['VX']['new']['diff']:.1f}" if "new" in TX["VX"] else PEND)   # for "cost X points"
five = [TX[v_].get("all") for v_ in ("V25", "V50", "V75", "V100")]   # wrong objects of the same relation
vc_all = TX["VC"].get("all")
num("TrImplAllAbs", f"{TX['VX']['all']['diff']:.1f}" if "all" in TX["VX"] else PEND)
if all(five) and vc_all:
    num("TrFourMin", min(v["diff"] for v in five), sign=True, nd=2); num("TrFourMax", max(v["diff"] for v in five), sign=True, nd=2)
    num("TrContraGain", -vc_all["diff"], nd=2)
    num("TrContraGainLo", -vc_all["hi90"], nd=2); num("TrContraGainHi", -vc_all["lo90"], nd=2)
else:
    for k in ("TrFourMin", "TrFourMax", "TrContraGain", "TrContraGainLo", "TrContraGainHi"):
        num(k, PEND)
for g in ("all", "new", "seven"):                 # the six variants together, for one sentence of 6.4
    vs = [TX[v_].get(g) for v_ in VLAB]
    if all(vs):
        num(f"TrVarMin{GN[g]}", min(v["diff"] for v in vs), sign=True, nd=2)
        num(f"TrVarMax{GN[g]}", max(v["diff"] for v in vs), sign=True, nd=2)
        num(f"TrVarPos{GN[g]}", words(sum(v["lo"] > 0 for v in vs)))
        num(f"TrVarNeg{GN[g]}", words(sum(v["hi"] < 0 for v in vs)))
        num(f"TrVarEquiv{GN[g]}", words(sum(v["equiv"] for v in vs)))
        num(f"TrVarBelow{GN[g]}", words(sum(v["lo90"] <= -1.5 for v in vs)))
        num(f"TrVarAbove{GN[g]}", words(sum(v["hi90"] >= 1.5 for v in vs)))
    else:
        for k in ("Min", "Max", "Pos", "Neg", "Equiv", "Below", "Above"):
            num(f"TrVar{k}{GN[g]}", PEND)
rows = [("\\emph{Matched format}", None, None),
        ("QA(S) $-$ D(S)rep", FX, "primary"), ("QA(S) $-$ D(S)", FX, "qa_vs_d"),
        ("QA(S) $-$ D(S)rep, cloze", FX, "primary_cloze"), ("MIX(KG-D) $-$ D(S)rep", FX, "more_facts"),
        ("\\emph{Multiple choice}", None, None),
        ("Repair channel", MX, "Repair channel"), ("Selection channel", MX, "Selection channel"),
        ("Both channels", MX, "Both channels"), ("Truth, declarative", MX, "Truth (declarative)"),
        ("Truth, question-answer", MX, "Truth (question-answer)"), ("QA $-$ declarative", MX, "Format (QA - declarative)"),
        ("\\emph{True minus variant}", None, None),
        ("25\\% false lines", TX, "V25"), ("50\\% false lines", TX, "V50"), ("75\\% false lines", TX, "V75"),
        ("all lines false", TX, "V100"), ("implausible objects", TX, "VX"), ("contradictions", TX, "VC"),
        ("slope per 100\\% false", TX, "slope")]
tb = ["\\begin{tabular}{@{}lccc@{}}", "\\toprule", "Contrast & 1B models & 7B models & further families \\\\"]
for lab, res, key in rows:
    if res is None:
        tb += ["\\midrule", f"\\multicolumn{{4}}{{@{{}}l}}{{{lab}}} \\\\"]
    else:
        tb.append(f"{lab} & " + " & ".join(rx_cell(res[key], g) for g in ("small", "large", "new")) + " \\\\")
(TAB / "tableB_controls.tex").write_text("\n".join(tb + ["\\bottomrule", "\\end{tabular}"]) + "\n")

fx_small = [RX["fmt"]["cells"].get(f"{t}|{b_}|primary") for t in SMALL for b_ in BUD]
import re                                                         # noqa: E402
N07 = dict(re.findall(r"\\newcommand\{\\N(\w+)\}\{([^}]*)\}", (TAB / "numbers.tex").read_text()))
mech4 = {k: float(N07[k]) for k in ("MechCfourQwenSeventy", "MechCfourQwenFifty", "MechCfourOlmoSeventy", "MechCfourOlmoFifty")}
fx_paper = [RX["fmt"]["cells"].get(f"{t}|{b_}|primary") for t in ("qwen15b", "olmo1b", "qwen7b", "olmo7b") for b_ in BUD]
num("FqNewQuestion", FQ["pooled"]["new"]["question"]["diff"], sign=True)
GUARDS_8K = [
    ("new" not in TX["VX"] or (TX["VX"]["new"]["lo"] > 0 and TX["VX"]["new"]["diff"] > TX["V100"]["new"]["diff"]),
     "abstract, 1, 6.4, 7 'objects of another relation and type cost Llama, Mistral and Phi-4, more than same-relation "
     "wrong objects'"),
    ("seven" not in TX["V100"] or (TX["V100"]["seven"]["diff"] < 1 and STATS["truth"]["pooled"]["all"]["diff"] < 1
                                   and ("seven" not in MX["truth_pooled"] or MX["truth_pooled"]["seven"]["diff"] < 1)),
     "abstract 'wrong objects of the same relation lower retention by less than one point on average, also under "
     "likelihood scoring'"),
    ("new" not in FX["primary"] or (FX["primary"]["new"]["diff"] < 0 and FX["primary"]["new"]["sig_neg"] == 2 and
                                   FQ["pooled"]["new"]["question"]["diff"] < FX["primary"]["new"]["diff"] and
                                   FX["more_facts"]["new"]["lo"] > 0),
     "6.4 'further families: QA keeps fewer facts at matched facts (two of six significant), less than at a fixed budget; "
     "more facts help'"),
    ("new" not in FX["interaction"] or all(FX["interaction"][g]["lo"] > 0 for g in ("small", "large", "new")),
     "6.4 'in every group QA text does relatively better with question than with cloze queries at matched facts'"),
    (not (all(five) and vc_all) or (all(v["equiv"] for v in five) and not vc_all["equiv"] and vc_all["hi"] < 0),
     "6.4 'four same-relation variants equivalent on Qwen2.5 and OLMo-2; contradictions keep significantly more facts'"),
    ("all" not in TX["VX"] or "new" not in TX["VX"] or (TX["VX"]["all"]["diff"] > TX["V100"]["all"]["diff"] and
                                                       TX["VX"]["new"]["diff"] > TX["V100"]["new"]["diff"]),
     "1 and 6.4 'objects of another relation and type cost more than same-relation wrong objects in both groups'"),
    ("all" not in TX["slope"] or not TX["slope"]["all"]["sig"], "6.4 'retention does not fall with the share of false lines' (pooled)"),
    ("seven" not in MX["Both channels"] or MX["Both channels"]["seven"]["sig_pos"] == 14,
     "6.2 'under multiple choice both channels beat C4/C4 in all 14 settings'"),
    ("seven" not in MX["truth_pooled"] or (MX["truth_pooled"]["seven"]["equiv"] and MX["truth_pooled"]["seven"]["diff"] > 0),
     "6.4 'under likelihood scoring true text keeps slightly more on all seven models, 90% interval inside the margin'"),
    (not MX["rule"] or all(r_["findings_survive"] for r_ in MX["rule"].values()),
     "App. B 'the channel results survive likelihood scoring by the rules fixed in advance (6 of 8, 5 of 6)'"),
    (any(c is None for c in fx_paper) or (all(c["diff"] > 0 for c in fx_paper) and not FX["rule"]["small"]["format_effect"]
                                         and not FX["rule"]["large"]["format_effect"]),
     "6.4 'matched format: a gain in all eight settings of Qwen2.5 and OLMo-2; the all-four criterion is met in neither group'"),
    ("large" not in FX["more_facts"] or (FX["more_facts"]["large"]["lo"] > 0 and FX["more_facts"]["small"]["diff"] < 0),
     "6.4 'more declarative facts add points on the 7B models but not on the 1B models'"),
    ("all" not in FX["primary_cloze"] or not FX["primary_cloze"]["all"]["sig"], "6.4 'with cloze queries the matched gain vanishes'"),
    (min(mech4, key=mech4.get) == "MechCfourQwenFifty" and max(mech4, key=mech4.get) == "MechCfourOlmoSeventy",
     "6.3 'C4 repair leaves NMechCfourQwenFifty to NMechCfourOlmoSeventy of the discarded error on the two 1B models'"),
    (any(c is None for c in fx_small) or (all(c["diff"] > 0 for c in fx_small) and sum(c["sig"] for c in fx_small) == 3
                                         and not FX["rule"]["small"]["format_effect"]),
     "6.4 'matched format on the 1B models: positive in all four settings, significant in three, one short of the criterion'"),
]
for ok, sentence in GUARDS_8K:
    assert ok, f"text no longer matches the data, rewrite: {sentence}"

# ---------------------------------------------------------------- 8h. guards for section 8
GUARDS_8 = [
    (all(v["sig"] and v["diff"] > 0 for v in eqa["Both channels"].values()), "6.5 'transfer: both channels keep more facts in every setting'"),
    (set(cf4_eq_neg) == {("olmo7b", "0.7"), ("olmo7b", "0.5"), ("mistral7b", "0.7"), ("mistral7b", "0.5")},
     "6.5 'C4 repair is again significantly worse than no repair on OLMo-2-7B and Mistral-7B at both budgets'"),
    (eq_paper_equiv, "6.5 'equivalent within the pre-registered 1.5 points' (paper models, EntityQuestions)"),
    (bool(POOLEQ.get("new|all", {}).get("equivalent_within_margin")),
     "6.5 'the further families: 90% interval inside the same margin' (EntityQuestions)"),
    (mix_none_qualifies, "6.2 'no share below one half keeps three quarters of the gain in six of eight settings'"),
    (mix_gain_increasing and mix_hs_increasing, "6.2 'on average, larger KG shares keep more facts and cost more HellaSwag accuracy'"),
    (all(v["both"]["sig"] and v["both"]["diff"] > 0 for v in SEQ.values()), "6.2/App. sequential 'both channels still beat C4/C4'"),
    (sum(v["rep"]["diff"] > 0 for v in SEQ.values()) < len(SEQ) or len(SEQ) < 5,
     "App. sequential: the text reports the repair channel as a count ('in X of Y settings'), not as 'in every setting'"),
    (all(v["both"]["diff"] > 0 for v in OBS.values()), "App. OBS 'both channels beat C4/C4'"),
    (human_order[0] == "judge" and acc["judge"] > acc["alias"], "7 'the Phi-4 judge agrees best with the human label'"),
    (a_fn > 3 * a_fp, "App. J 'the alias rule errs mostly by missing correct answers'"),
    (all(not (v["truth"]["sig"] and v["truth"]["diff"] < 0) for v in SEQ.values()), "App. sequential 'no setting significantly in favour of counterfactual text'"),
    (all(v["sig"] and v["diff"] > 0 for v in sub["subject_disjoint"]), "App. N '... subject-disjoint subset: both channels in every setting'"),
    (all(v["diff"] > 0 for v in sub["answer_disjoint"]), "App. N 'answer-disjoint subset: positive in every setting'"),
    (ov["subject_either"]["questions"] == ov["questions"] - ov["subject_disjoint_questions"], "App. N overlap counts add up"),
    (ov["subject_is_calibration_entity"]["as_subject_name"] == 0, "App. N 'the 53 questions match answer aliases, not calibration subjects'"),
    (all(v["diff"] > 0 for v in sub["both_disjoint"]), "App. N 'without either kind of overlap: positive in every setting'"),
    (sorted(own) == sorted([f"{t}|{b_}" for t in SMALL for b_ in BUD] + ["olmo7b|0.5"]),
     "6.4/App. K 'format itself matters on the four 1B settings and on OLMo-2-7B at 50%'"),
    (sorted(cov_only) == ["olmo7b|0.7", "qwen7b|0.5", "qwen7b|0.7"], "6.4/App. K 'within reach of coverage in the other three 7B settings'"),
    (max(small_b) < min(bd[f"{t}|{b_}"]["format_contrast"] for t in SMALL for b_ in BUD), "App. K 'the gain exceeds the coverage bound on the 1B models'"),
    (tr_holm <= 1 and HOLM["question|paper|Anchoring"]["holm_pos"] == 0, "App. statistics 'single-setting truth and anchoring gains do not survive Holm'"),
]
for ok, sentence in GUARDS_8:
    assert ok, f"text no longer matches the data, rewrite: {sentence}"
