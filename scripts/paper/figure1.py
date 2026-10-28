"""Figure 1 as an editable draw.io file: outputs/figures/figure1.drawio (plus a PNG preview and a vector PDF).

(a) Lemma 1 drawn as a regression picture: the repair is a least-squares fit on the calibration text. Fitted on Web
    text it misses the contexts of knowledge questions; the gap to the best fit for those contexts is the mismatch term.
    The clouds are a schematic; every number printed under the plot is measured.
(b) A minimal-pair ledger: real calibration lines written from the knowledge graph, each changing one part of the
    same fact (highlighted), with the measured change in fact retention relative to the true statements (mean over the
    eight model-budget settings, one small dot per setting).

Sources: sentences from data/calib and data/calib_paired (checked to exist verbatim), retention from results/merged,
panel (a) numbers from outputs/tables/paper_numbers.json (scripts/analysis/paper_models.py).

python scripts/paper/figure1.py [--render]
--render needs Google Chrome (headless), Ghostscript, network access for the draw.io viewer script, and the Linux
Libertine and Biolinum OpenType fonts (found with kpsewhich, or set LIBERTINE_DIR to their folder).
"""
import html
import json
import os
import subprocess
import sys
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from kgcal import paths as P            # noqa: E402

RES, FIG, TAB = P.MERGED, P.FIGURES, P.TABLES
TAGS = ["qwen15b", "olmo1b", "qwen7b", "olmo7b"]
BUDGETS = ["0.7", "0.5"]

# ---------------------------------------------------------------- data
def bits(tag):
    raw = json.loads((RES / f"{tag}_bits.json").read_text())
    return {k: {f: np.array([c == "1" for c in v[f]], bool) for f in v} for k, v in raw.items()}


B = {t: bits(t) for t in TAGS}
DENSE = {t: B[t]["1.0|dense"]["test_q"] for t in TAGS}
COND = {"C4": "C4|C4", "KG-D": "MIX:KG-D|MIX:KG-D", "KG-D~": "MIX:KG-D~|MIX:KG-D~", "KG-S": "MIX:KG-S|MIX:KG-S",
        "KG-QA": "MIX:KG-QA|MIX:KG-QA", "KG-QA~": "MIX:KG-QA~|MIX:KG-QA~", "KG-O": "MIX:KG-O2|MIX:KG-O2",
        "KG-A": "MIX:KG-Ac|MIX:KG-Ac", "WIKI": "MIX:WIKI|MIX:WIKI"}


def retention(tag, budget, src):
    return 100 * B[tag][f"{budget}|{COND[src]}"]["test_q"][DENSE[tag]].mean()


def effect(src):
    """Retention minus retention with the true statements (KG-D), for each of the eight settings."""
    return np.array([retention(t, b, src) - retention(t, b, "KG-D") for t in TAGS for b in BUDGETS])


NUM = json.loads((TAB / "paper_numbers.json").read_text())
num = lambda k: NUM[k].replace("\\%", "%")
mech_c4 = [float(NUM[k]) for k in ("MechCfourQwenSeventy", "MechCfourQwenFifty", "MechCfourOlmoSeventy", "MechCfourOlmoFifty")]


def has_line(path, line):
    return any(json.loads(l) == line for l in open(path))


# held-out probe shown in panel (a): lost after C4 calibration, kept after KG calibration, per model at 70%
from kgcal import data as D  # noqa: E402
rows = D.load_popqa()
TEST = D.facts_by_split(rows, D.make_split(rows))["test"]
probe = next(i for i, r in enumerate(TEST) if r["subj"] == "Kagemusha" and r["prop"] == "director")
probe_c4 = [int(B[t]["0.7|C4|C4"]["test_q"][probe]) for t in TAGS]
probe_kg = [int(B[t]["0.7|MIX:KG-D|MIX:KG-D"]["test_q"][probe]) for t in TAGS]
probe_dense = [int(DENSE[t][probe]) for t in TAGS]
assert probe_dense == [1, 1, 1, 1] and probe_c4 == [0, 0, 0, 0] and probe_kg == [1, 1, 1, 1], "probe text must change"

# ---------------------------------------------------------------- drawing primitives (draw.io XML)
SANS = "Linux Biolinum O"
SERIF = "Linux Libertine O"
INK, MUTED, FAINT, RULE = "#1f2937", "#6b7280", "#9ca3af", "#d1d5db"
COL = {"C4": "#7f7f7f", "WIKI": "#8c564b", "KG-D": "#1f77b4", "KG-D~": "#ff7f0e", "KG-QA": "#2ca02c",
       "KG-QA~": "#5fa84f", "KG-S": "#9467bd", "KG-A": "#d62728", "KG-O": "#17becf"}
# highlighter tint by the property a line changes (the same property always has the same tint)
TINT = {"truth": "#ffe2c4", "subject": "#e9ddf5", "form": "#d4eecd", "relation": "#c9eef3", "entity": "#f9d4d4",
        "prose": "#ecdcd5"}
cells = []


def _id():
    return f"n{len(cells) + 2}"


def vertex(value, x, y, w, h, style):
    i = _id()
    cells.append(f'<mxCell id="{i}" value="{escape(value, {chr(34): "&quot;"})}" style="{style}" vertex="1" parent="1">'
                 f'<mxGeometry x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" as="geometry"/></mxCell>')
    return i


def text(s, x, y, w, h, size=14, color=INK, font=SANS, align="left", bold=False, italic=False, rot=0, valign="middle"):
    fs = (1 if bold else 0) + (2 if italic else 0)
    style = (f"text;html=1;whiteSpace=nowrap;overflow=visible;align={align};verticalAlign={valign};fontFamily={font};"
             f"fontSize={size};fontColor={color};fontStyle={fs};spacing=0;spacingLeft=0;spacingRight=0;"
             f"strokeColor=none;fillColor=none;" + (f"rotation={rot};" if rot else ""))
    return vertex(s, x, y, w, h, style)


def line(x1, y1, x2, y2, color=INK, width=1.0, dash=None, start="none", end="none", opacity=100):
    i = _id()
    style = (f"endArrow={end};startArrow={start};html=1;rounded=0;strokeColor={color};strokeWidth={width};"
             f"endSize=5;startSize=5;opacity={opacity};" + (f"dashed=1;dashPattern={dash};" if dash else ""))
    cells.append(f'<mxCell id="{i}" value="" style="{style}" edge="1" parent="1"><mxGeometry relative="1" as="geometry">'
                 f'<mxPoint x="{x1:.1f}" y="{y1:.1f}" as="sourcePoint"/><mxPoint x="{x2:.1f}" y="{y2:.1f}" as="targetPoint"/>'
                 f'</mxGeometry></mxCell>')
    return i


def dot(x, y, r, color, opacity=100, stroke="none"):
    return vertex("", x - r, y - r, 2 * r, 2 * r,
                  f"ellipse;fillColor={color};strokeColor={stroke};opacity={opacity};strokeWidth=1;")


def rect(x, y, w, h, fill, stroke="none", rounded=0):
    return vertex("", x, y, w, h, f"rounded={1 if rounded else 0};arcSize={rounded};fillColor={fill};strokeColor={stroke};")


def hl(s, kind):
    return f'<span style="background-color:{TINT[kind]};border-radius:2px;">{html.escape(s)}</span>'


def mark(ok):
    return '<span style="color:#2c7a3f;">&#10003;</span>' if ok else '<span style="color:#b42318;">&#10007;</span>'


# ---------------------------------------------------------------- panel (a): repair is a fit on the calibration text
AX0, AY0, AX1, AY1 = 48, 236, 342, 42            # plot box in px (left, bottom, right, top)
X = lambda u: AX0 + (AX1 - AX0) * u / 10.0       # data x in [0, 10]
Y = lambda v: AY0 - (AY0 - AY1) * (v + 1) / 11.0  # data y in [-1, 10]

text("<b>(a)</b>&nbsp; Repair fitted on Web text misses knowledge contexts", 0, 0, 356, 20, size=15)
rng = np.random.default_rng(7)
web_x = rng.uniform(0.7, 4.6, 26); web_y = 0.95 * web_x + 0.9 + rng.normal(0, 0.42, 26)
kq_x = rng.uniform(5.7, 9.5, 22); kq_y = 0.22 * kq_x + 2.6 + rng.normal(0, 0.38, 22)
a_c4 = np.polyfit(web_x, web_y, 1)                                    # repair fitted on Web text
a_p = np.polyfit(kq_x, kq_y, 1)                                       # best linear fit for knowledge questions
f_c4, f_p = (lambda u: float(np.polyval(a_c4, u))), (lambda u: float(np.polyval(a_p, u)))
line(AX0, AY0, AX1 + 4, AY0, color=FAINT, width=1)                    # axes
line(AX0, AY0, AX0, AY1 - 4, color=FAINT, width=1)
text("activity of the kept units", AX0, AY0 + 3, AX1 - AX0, 16, size=13, color=MUTED, italic=True, align="center")
text("output of the removed units", AX0 - 99, (AY0 + AY1) / 2 - 8, 176, 16, size=13, color=MUTED, italic=True,
     align="center", rot=-90)
text("schematic", X(9.9) - 70, Y(-0.5) - 7, 70, 14, size=12, color=FAINT, italic=True, align="right")
text(f'error on knowledge questions = <i style="color:{MUTED};">unavoidable</i> + '
     f'<b style="color:#b42318;">mismatch</b>', X(0.2), Y(9.85) - 7, 300, 14, size=12, color=INK)
line(X(0.15), Y(0), X(9.95), Y(0), color=FAINT, width=1.2, dash="2 3")  # no repair: nothing for the removed units
text("no repair", X(0.3), Y(0) - 16, 70, 14, size=12, color=FAINT, italic=True)
for x_, y_ in zip(web_x, web_y):
    dot(X(x_), Y(y_), 3.2, COL["C4"], opacity=55)
for x_, y_ in zip(kq_x, kq_y):
    dot(X(x_), Y(y_), 3.2, COL["KG-D"], opacity=75)
line(X(0.3), Y(f_c4(0.3)), X(9.3), Y(f_c4(9.3)), color="#4b5563", width=1.8)
line(X(5.45), Y(f_p(5.45)), X(9.85), Y(f_p(9.85)), color=COL["KG-D"], width=1.4)
text("C4 repair", X(8.12), Y(7.55) - 7, 64, 14, size=12, color="#4b5563", bold=True)
text("best fit", X(5.3) - 60, Y(f_p(5.3)) + 2, 60, 14, size=12, color=COL["KG-D"], italic=True, align="right")
xm = 7.9                                                               # Lemma 1: mismatch = deployed fit vs best fit
line(X(xm), Y(f_c4(xm)) + 3, X(xm), Y(f_p(xm)) - 3, color="#b42318", width=1.5, start="classic", end="classic")
xb = 9.72                                                              # Lemma 1: scatter no linear fit can remove
sd = float(np.std(kq_y - np.polyval(a_p, kq_x)))
line(X(xb), Y(f_p(xb) + 1.6 * sd), X(xb), Y(f_p(xb) - 1.6 * sd), color=MUTED, width=1, start="none", end="none")
for sgn in (1, -1):
    line(X(xb) - 3, Y(f_p(xb) + sgn * 1.6 * sd), X(xb) + 1, Y(f_p(xb) + sgn * 1.6 * sd), color=MUTED, width=1)
text("Web text (C4)", X(0.4), Y(7.95) - 7, 110, 14, size=12, color="#4b5563", bold=True)
text("&ldquo;Conduct effective product control &hellip;&rdquo;", X(0.4), Y(7.1) - 7, 170, 14, size=12, color=MUTED,
     font=SERIF, italic=True)
text("knowledge questions", X(9.9) - 120, Y(2.45) - 7, 120, 14, size=12, color=COL["KG-D"], bold=True, align="right")
text("&ldquo;Q: Who directed Kagemusha? A:&rdquo;", X(9.9) - 180, Y(1.55) - 7, 180, 14, size=12, color=MUTED,
     font=SERIF, italic=True, align="right")

# measured, under the plot
yb = 262
text("measured: error left on knowledge questions (1 = no repair)", 2, yb, 350, 14, size=12, color=MUTED, italic=True)
BX0, BX1 = 116, 292
bx = lambda r: BX0 + (BX1 - BX0) * r / 1.1
for k, (lab, lo, hi, c, light) in enumerate([
        ("C4 repair", min(mech_c4), max(mech_c4), "#4b5563", "#c4c8ce"),
        ("C4 + KG repair", float(NUM["MechMixLo"]), float(NUM["MechMixHi"]), COL["KG-D"], "#b9d3ea")]):
    y = yb + 26 + 20 * k
    text(lab, 2, y - 8, 108, 16, size=13, color=c, bold=True, align="right")
    rect(bx(0), y - 5, bx(lo) - bx(0), 10, light)
    rect(bx(lo), y - 5, bx(hi) - bx(lo), 10, c)
    text(f"{lo:.2f}&ndash;{hi:.2f}", bx(1.1) + 6, y - 8, 70, 16, size=13, color=c)
line(bx(1.0), yb + 16, bx(1.0), yb + 56, color=INK, width=1, dash="2 2")
text("&ldquo;Who directed Kagemusha?&rdquo; held out; one mark per model", 2, yb + 66, 350, 14, size=12,
     color=MUTED, italic=True)
text(f'<b style="color:#4b5563;">C4 calibration</b>&nbsp; {"&thinsp;".join(mark(v) for v in probe_c4)}'
     f'&emsp;&emsp;<b style="color:{COL["KG-D"]};">KG calibration</b>&nbsp; {"&thinsp;".join(mark(v) for v in probe_kg)}',
     2, yb + 84, 350, 16, size=13, color=INK)

# ---------------------------------------------------------------- panel (b): the minimal-pair ledger
line(360, 6, 360, 376, color=RULE, width=1)
PX = 372                                          # left edge of panel (b)
CHIP_R = 462                                      # right edge of the chip / note column
SX = 474                                          # sentence column
DX0, DX1, LO, HI = 834, 958, -6.0, 8.0            # dot plot
VX = 998                                          # right edge of the value column
dxp = lambda v: DX0 + (DX1 - DX0) * (v - LO) / (HI - LO)
text("<b>(b)</b>&nbsp; Facts follow form and coverage, not truth", PX, 0, 600, 20, size=15)
text("one line from each calibration text", SX, 25, 330, 14, size=12, color=MUTED, italic=True)
text("retention vs. true line (points)", DX0 - 8, 25, VX - DX0 + 8, 14, size=12, color=MUTED, italic=True,
     align="center")
LEDGER = [
    ("KG-D", "true", P.calib_file("KG-D"), "Little Malcolm was directed by Stuart Cooper.",
     "Little Malcolm was directed by Stuart Cooper."),
    ("KG-D~", "wrong object", P.calib_file("KG-D~"), "Little Malcolm was directed by Shaji Kailas.",
     f"Little Malcolm was directed by {hl('Shaji Kailas', 'truth')}."),
    ("KG-S", "made-up subject", P.calib_file("KG-S"), "Private One was directed by Stuart Cooper.",
     f"{hl('Private One', 'subject')} was directed by Stuart Cooper."),
    ("KG-QA", "question form", P.calib_file("KG-QA"), "Q: Who was the director of Little Malcolm? A: Stuart Cooper",
     f"{hl('Q: Who was the director of', 'form')} Little Malcolm{hl('? A:', 'form')} Stuart Cooper"),
    ("KG-QA~", "+ wrong object", P.calib_file("KG-QA~"), "Q: Who was the director of Little Malcolm? A: Hobart Henley",
     f"Q: Who was the director of Little Malcolm? A: {hl('Hobart Henley', 'truth')}"),
    ("KG-O", "other relation", P.CALIB / "KG-O2.jsonl", "The cast member of Little Malcolm is John Hurt.",
     f"The {hl('cast member', 'relation')} of Little Malcolm is {hl('John Hurt', 'relation')}."),
    ("KG-A", "held-out entity", P.CALIB / "KG-Ac.jsonl", "The cast member of Kagemusha is Tsutomu Yamazaki.",
     f"The cast member of {hl('Kagemusha', 'entity')} is Tsutomu Yamazaki."),
    ("WIKI", "Wikipedia prose", P.CALIB / "WIKI.jsonl", None,
     hl("Little Malcolm is a 1974 British comedy drama film", "prose") + " &hellip;"),
    ("C4", "Web text", None, None,
     f'<span style="color:{MUTED};">Conduct effective product control and valuation &hellip;</span>'),
]
wiki_line = next(json.loads(l) for l in open(P.CALIB / "WIKI.jsonl") if json.loads(l).startswith("Little Malcolm is a 1974"))
assert wiki_line.startswith("Little Malcolm is a 1974 British comedy drama film directed by Stuart Cooper")
for src, note, path, raw, shown in LEDGER:
    if raw is not None:
        assert has_line(path, raw), (src, raw)

Y0, PITCH, GAP = 56, 32.0, 9
ys = {}
for k, (src, note, path, raw, shown) in enumerate(LEDGER):
    y = Y0 + PITCH * k + GAP * ((k >= 3) + (k >= 5) + (k >= 8)) + 7 * (k >= 1)   # family gaps; room for the triple
    ys[src] = y
    text(src.replace("~", "&#8202;~"), CHIP_R - 90, y - 15, 90, 16, size=13, color=COL[src], bold=True, align="right")
    text(note, CHIP_R - 100, y + 1, 100, 14, size=12, color=MUTED, italic=True, align="right")
    text(shown, SX, y - 10, 360, 18, size=13.5, font=SERIF, color=INK)
text("triple (Little Malcolm, director, Stuart Cooper)", SX, ys["KG-D"] + 8, 330, 14, size=12, color=FAINT,
     italic=True)
for top, bottom in (("KG-D", "KG-S"), ("KG-QA", "KG-QA~")):          # paired: written line by line from the true line
    xk = CHIP_R + 5
    line(xk, ys[top] - 1, xk, ys[bottom], color=FAINT, width=1)
    for s in [r[0] for r in LEDGER]:
        if ys[top] < ys[s] <= ys[bottom]:
            line(xk, ys[s], xk + 4, ys[s], color=FAINT, width=1)
ybot = ys["C4"] + 16
for t in (-4, 4, 8):
    line(dxp(t), Y0 - 18, dxp(t), ybot, color="#eceef1", width=1)
line(dxp(0), Y0 - 18, dxp(0), ybot, color=INK, width=1)
for t in (-4, 0, 4, 8):
    text("&minus;4" if t == -4 else (f"+{t}" if t else "0"), dxp(t) - 15, ybot + 2, 30, 14, size=12, color=MUTED,
         align="center")
for src, *_ in LEDGER:
    y = ys[src]
    if src == "KG-D":
        dot(dxp(0), y, 4.4, COL[src])
        text("0", VX - 34, y - 8, 34, 16, size=13, color=INK, align="right")
        continue
    e = effect(src)
    val = f"{e.mean():+.1f}".replace("-", "&minus;")
    if src == "C4":
        line(dxp(-1.2), y, dxp(LO) - 4, y, color=COL[src], width=1.8, end="block")
        text(val, VX - 40, y - 8, 40, 16, size=13, color=COL[src], bold=True, align="right")
        continue
    for v in e:
        dot(dxp(v), y, 2.1, COL[src], opacity=38)
    dot(dxp(e.mean()), y, 4.4, COL[src])
    text(val, VX - 34, y - 8, 34, 16, size=13, color=INK, align="right")

# ---------------------------------------------------------------- write
xml = ('<mxfile host="app.diagrams.net" type="device"><diagram id="figure1" name="Figure 1">'
       '<mxGraphModel dx="1000" dy="390" grid="0" gridSize="5" guides="1" tooltips="1" connect="0" arrows="0" fold="1" '
       'page="0" pageScale="1" pageWidth="1000" pageHeight="390" math="0" shadow="0" background="#ffffff"><root>'
       '<mxCell id="0"/><mxCell id="1" parent="0"/>' + "".join(cells) + "</root></mxGraphModel></diagram></mxfile>")
out = FIG / "figure1.drawio"
out.write_text(xml)
print(f"wrote {out} ({len(cells)} cells)")
for src, *_ in LEDGER:
    if src not in ("KG-D", "C4"):
        e = effect(src); print(f"  {src:7s} {e.mean():+.1f}  [{e.min():+.1f}, {e.max():+.1f}]")

# ---------------------------------------------------------------- render (draw.io viewer in headless Chrome)
if "--render" in sys.argv:
    tt = os.environ.get("LIBERTINE_DIR") or str(Path(subprocess.run(["kpsewhich", "LinLibertine_R.otf"], capture_output=True,
                                                                      text=True).stdout.strip()).parent)
    tt = Path(tt)
    face = lambda fam, f, w, s: (f"@font-face{{font-family:'{fam}';src:url('file://{tt / f}');font-weight:{w};"
                                 f"font-style:{s};}}")
    css = "".join([face(SERIF, "LinLibertine_R.otf", "normal", "normal"), face(SERIF, "LinLibertine_RB.otf", "bold", "normal"),
                   face(SERIF, "LinLibertine_RI.otf", "normal", "italic"), face(SANS, "LinBiolinum_R.otf", "normal", "normal"),
                   face(SANS, "LinBiolinum_RB.otf", "bold", "normal"), face(SANS, "LinBiolinum_RI.otf", "normal", "italic")])
    cfg = json.dumps({"highlight": "none", "nav": False, "toolbar": "", "lightbox": False, "xml": xml})
    page = FIG / ".figure1_view.html"
    page.write_text(f'<html><head><style>{css} body{{margin:0;background:#fff}}</style></head><body>'
                    f'<div class="mxgraph" style="max-width:100%" data-mxgraph="{html.escape(cfg)}"></div>'
                    f'<script src="https://viewer.diagrams.net/js/viewer-static.min.js"></script></body></html>')
    png = FIG / "figure1_drawio_preview.png"
    common = ["google-chrome", "--headless=new", "--no-sandbox", "--disable-gpu", "--hide-scrollbars",
              "--allow-file-access-from-files", "--virtual-time-budget=20000"]
    subprocess.run(common + ["--force-device-scale-factor=2", "--window-size=1060,760", f"--screenshot={png}",
                             f"file://{page}"], capture_output=True, timeout=120)
    from PIL import Image, ImageChops
    im = Image.open(png).convert("RGB")
    bbox = ImageChops.difference(im, Image.new("RGB", im.size, "white")).getbbox()
    im.crop((max(bbox[0] - 12, 0), max(bbox[1] - 12, 0), bbox[2] + 12, bbox[3] + 12)).save(png)
    print("preview:", png, Image.open(png).size)
    # vector PDF: a print page as large as the drawing, then a tight crop with Ghostscript
    ppage = FIG / ".figure1_print.html"
    ppage.write_text(f'<html><head><style>{css} @page{{size:1100px 460px;margin:0}} body{{margin:0;background:#fff}}'
                     f'</style></head><body><div class="mxgraph" style="width:1080px" data-mxgraph="{html.escape(cfg)}">'
                     f'</div><script src="https://viewer.diagrams.net/js/viewer-static.min.js"></script></body></html>')
    raw = FIG / ".figure1_raw.pdf"
    subprocess.run(common + ["--no-pdf-header-footer", f"--print-to-pdf={raw}", f"file://{ppage}"], capture_output=True,
                   timeout=120)
    bb = subprocess.run(["gs", "-q", "-dBATCH", "-dNOPAUSE", "-sDEVICE=bbox", str(raw)], capture_output=True, text=True).stderr
    x0, y0, x1, y1 = [float(v) for v in bb.split("%%HiResBoundingBox:")[1].split()[:4]]
    x0, y0, x1, y1 = x0 - 2, y0 - 2, x1 + 2, y1 + 2
    pdf = FIG / "figure1.pdf"
    subprocess.run(["gs", "-q", "-dBATCH", "-dNOPAUSE", "-sDEVICE=pdfwrite", "-dCompatibilityLevel=1.5", f"-o{pdf}",
                    f"-dDEVICEWIDTHPOINTS={x1 - x0:.2f}", f"-dDEVICEHEIGHTPOINTS={y1 - y0:.2f}", "-dFIXEDMEDIA",
                    "-c", f"<</PageOffset [{-x0:.2f} {-y0:.2f}]>> setpagedevice", "-f", str(raw)], check=True)
    for tmp in (raw, page, ppage):
        tmp.unlink(missing_ok=True)
    print("pdf:", pdf, f"{x1 - x0:.0f} x {y1 - y0:.0f} pt")
