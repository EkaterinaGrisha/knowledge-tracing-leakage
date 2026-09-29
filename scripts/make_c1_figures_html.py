"""Paper C1: figures authored as SVG and rendered through headless Chrome.

Why not matplotlib. These two figures are diagrams, not plots: one shows how three
rows of a table relate to each other, the other compares two numbers per dataset.
Matplotlib fights both — its text metrics are hard to control, its defaults look
like a notebook, and the bilingual labels the journal wants need precise spacing.
Hand-written SVG gives that control, and Chrome renders it to a 300 dpi PNG and a
vector PDF, which is exactly what the journal asks for.

Figure 1 — the mechanism, drawn as the data actually looks: one answer becomes three
table rows sharing one answer, and the measured AUC sits in the last column, so the
evidence and the diagram are the same object.

Figure 2 — one row per dataset, two dots joined by a line: the AUC of the clean path
and of the averaged path. Where nothing is unrolled the dots coincide, which is the
control stated visually rather than in words.

Every number is read from the artifacts; none is written into this file.

Geometry: the drawing is laid out in a 512-unit-wide space that renders to 13 cm at
300 dpi (the journal allows 14 cm). One unit is 0.72 pt on paper, so a font-size of
11 here prints at about 7.9 pt.

Outputs (figures/):
  c1_fig1_mechanism.{svg,png,pdf}
  c1_fig2_inflation.{svg,png,pdf}

Usage:
  python -m scripts.make_c1_figures_html
  python -m scripts.make_c1_figures_html --no-render   # only the .svg/.html
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from ktx import paths

CROSS = paths.ARTIFACTS_DIR / "cross_dataset"
FIG_DIR = paths.RESEARCH_DIR / "figures"
def _find_chrome() -> Path | None:
    """Найти безголовый браузер для рендера. Порядок: переменная окружения,
    затем обычные имена в PATH, затем стандартный путь macOS."""
    env = os.environ.get("CHROME_PATH")
    if env and Path(env).exists():
        return Path(env)
    for name in ("google-chrome", "google-chrome-stable", "chromium",
                 "chromium-browser", "chrome", "msedge"):
        found = shutil.which(name)
        if found:
            return Path(found)
    mac = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    return mac if mac.exists() else None


CHROME = _find_chrome()

W = 512                     # drawing units across = 13 cm at 300 dpi
PAD = 8                     # keep content off the very edge of the image
SCALE = 3                   # css px -> device px, so 512 * 3 = 1536 px

INK = "#1B2129"
MUTED = "#6E7883"
RULE = "#C9D0D8"
LEAK = "#A8403A"
LEAK_BG = "#FAEDEB"
# 2026-09-03: was #2F5D87, which fails the dataviz chroma floor (0.085 against a floor
# of 0.1) and therefore reads gray rather than blue. #215D9E is the nearest step that
# passes every check against #A8403A: chroma floor, CVD separation (protan dE 16.0),
# normal-vision separation and contrast on a light surface. The pale tints below are
# surfaces, not data colours, and are left as they are.
CLEAN = "#215D9E"
CLEAN_BG = "#EAF1F7"
PANEL = "#F4F6F8"

DISPLAY = {
    "assist2009": "ASSISTments-2009",
    "assist2012": "ASSISTments-2012",
    "assist2015": "ASSISTments-2015",
    "assist2017": "ASSISTments-2017",
    "algebra2005": "Algebra-2005",
    "bridge2algebra2006": "Bridge-to-Algebra-2006",
    "ednet": "EdNet-KT1-5k",
}

FONT = ("-apple-system, 'Helvetica Neue', Helvetica, Arial, "
        "'Liberation Sans', sans-serif")

EXAMPLE = ("ednet", "dkt")   # the worked example carried by figure 1


# ----------------------------------------------------------------- numbers

def figure1_numbers() -> dict[str, float]:
    rep = pd.read_csv(CROSS / "leakage_by_repeat_summary.csv")
    auc = pd.read_csv(CROSS / "leakage_auc_summary.csv")
    ds, model = EXAMPLE

    def pos(klass: str) -> float:
        row = rep[(rep.dataset == ds) & (rep.model == model)
                  & (rep.position_class == klass)]
        return float(row.auc_mean.iloc[0])

    cell = auc[(auc.dataset == ds) & (auc.model == model)].iloc[0]
    return {
        "first": pos("first_row"), "rep1": pos("repeat_1"), "rep2": pos("repeat_2"),
        "late": float(cell.late_mean_auc_mean), "clean": float(cell.pykt_question_auc_mean),
    }


def figure2_rows() -> list[dict]:
    matrix = pd.read_csv(CROSS / "leakage_auc_matrix.csv")
    wide = matrix.pivot_table(index=["dataset", "model", "fold"],
                              columns="source", values="auc").reset_index()
    wide = wide.dropna(subset=["pykt_question", "late_mean"])
    kcq = pd.read_csv(CROSS / "leakage_kc_per_question.csv")
    g = (wide.groupby("dataset")
         .agg(clean=("pykt_question", "mean"), late=("late_mean", "mean"))
         .reset_index().merge(kcq[["dataset", "kc_per_question"]], on="dataset"))
    g["gap"] = g.late - g.clean
    # Предсказание механизма: усреднение при условии, что строки-продолжения
    # воспроизводят показанный ответ. Ничего не подбирается, см.
    # leakage_model_prediction.py. Файла может не быть — тогда рисунок строится
    # без предсказания, как раньше.
    mp = CROSS / "leakage_model_prediction.csv"
    if mp.exists():
        pred = (pd.read_csv(mp).groupby("dataset").auc_predicted_avg.mean()
                .rename("predicted").reset_index())
        g = g.merge(pred, on="dataset", how="left")
    else:
        g["predicted"] = float("nan")
    # Вторичный ключ обязателен: три набора имеют кратность ровно 1.000, и без него
    # их порядок зависит от реализации сортировки, то есть рисунок перестаёт быть
    # воспроизводимым побайтово.
    g = g.sort_values(["kc_per_question", "dataset"], ascending=[False, True])
    return g.to_dict("records")


# -------------------------------------------------------------- svg helpers

def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, s, size=11, fill=INK, weight="400", anchor="start",
         style="normal", spacing=None) -> str:
    ls = f' letter-spacing="{spacing}"' if spacing else ""
    return (f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" '
            f'fill="{fill}" font-weight="{weight}" text-anchor="{anchor}" '
            f'font-style="{style}"{ls}>{esc(s)}</text>')


def rect(x, y, w, h, fill="none", stroke="none", sw=1, rx=0) -> str:
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}" rx="{rx}"/>')


def line(x1, y1, x2, y2, stroke=RULE, sw=1, dash=None) -> str:
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" '
            f'stroke-width="{sw}"{d}/>')


def path(d, stroke=INK, sw=1, fill="none", marker=True) -> str:
    m = ' marker-end="url(#arrow)"' if marker else ""
    return f'<path d="{d}" stroke="{stroke}" stroke-width="{sw}" fill="{fill}"{m}/>'


def defs() -> str:
    return ('<defs>'
            f'<marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" '
            f'markerWidth="5" markerHeight="5" orient="auto-start-reverse">'
            f'<path d="M 0 0 L 10 5 L 0 10 z" fill="context-stroke"/></marker>'
            f'<marker id="arrowleak" viewBox="0 0 10 10" refX="8" refY="5" '
            f'markerWidth="5" markerHeight="5" orient="auto-start-reverse">'
            f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{LEAK}"/></marker>'
            '</defs>')


def bilingual(x, y, ru, en, size=11, weight="600", anchor="start") -> str:
    return (text(x, y, ru, size=size, weight=weight, anchor=anchor)
            + text(x, y + size * 1.15, en, size=size - 1.5, fill=MUTED,
                   style="italic", anchor=anchor))


# ------------------------------------------------------------------ figure 1

def figure1_svg(n: dict[str, float]) -> tuple[str, int]:
    L, R = PAD, W - PAD
    o: list[str] = [defs()]

    # --- stage 1: the record as it is stored -------------------------------
    o.append(bilingual(L, 14, "1. Как ответ записан в данных",
                       "1. How the answer is stored"))
    o.append(rect(L, 30, R - L, 30, fill=PANEL, stroke=RULE))
    o.append(text(L + 12, 49, "задание Q17", size=12, weight="600"))
    o.append(text(L + 112, 49, "компоненты знания  c5, c8, c12", size=12))
    o.append(text(R - 12, 49, "ответ верный", size=12, anchor="end"))

    o.append(path(f"M {W/2} 62 L {W/2} 82", sw=1.2))
    o.append(text(W / 2 + 8, 70, "развёртка на компоненты", size=9.5, fill=MUTED))
    o.append(text(W / 2 + 8, 80, "unrolling into components", size=8.5, fill=MUTED,
                  style="italic"))

    # --- stage 2: the same answer as three rows ----------------------------
    top = 96
    o.append(bilingual(L, top, "2. Один ответ становится тремя строками",
                       "2. One answer becomes three rows"))

    th = 22                      # row height
    y0 = top + 34                # header baseline
    cols = [L + 4, L + 88, L + 202, L + 316]
    heads = [("позиция", "position"), ("задание · компонент", "question · component"),
             ("ответ", "answer"), ("is_repeat", "")]
    o.append(line(L, y0 + 16, R, y0 + 16, stroke=INK, sw=1.1))
    for x, (ru, en) in zip(cols, heads):
        o.append(text(x, y0, ru, size=9.5, fill=MUTED))
        if en:
            o.append(text(x, y0 + 10, en, size=8, fill=MUTED, style="italic"))
    o.append(text(R - 4, y0, "AUC модели", size=9.5, fill=MUTED, anchor="end"))
    o.append(text(R - 4, y0 + 10, "model AUC", size=8, fill=MUTED, style="italic",
                  anchor="end"))

    rows = [
        ("t",   "Q17 · c5",  "верный", "0", n["first"], False),
        ("t+1", "Q17 · c8",  "верный", "1", n["rep1"], True),
        ("t+2", "Q17 · c12", "верный", "1", n["rep2"], True),
    ]
    tbl = y0 + 16
    for i, (pos, qc, ans, rep, auc, leaked) in enumerate(rows):
        ry = tbl + i * th
        if leaked:
            o.append(rect(L, ry, R - L, th, fill=LEAK_BG))
        base = ry + 15
        o.append(text(cols[0], base, pos, size=11, fill=MUTED))
        o.append(text(cols[1], base, qc, size=11.5))
        o.append(text(cols[2], base, ans, size=11.5))
        o.append(text(cols[3], base, rep, size=11,
                      fill=LEAK if leaked else MUTED, weight="600" if leaked else "400"))
        o.append(text(R - 4, base, f"{auc:.3f}", size=12.5, anchor="end",
                      weight="700", fill=LEAK if leaked else INK))
        o.append(line(L, ry + th, R, ry + th, stroke=RULE))

    # the answer of row t reaching the context of rows t+1 and t+2; the arrows
    # live in the gap between the two columns, the wording goes under the table
    ax = cols[2] + 46
    o.append(f'<path d="M {ax} {tbl+15} C {ax+22} {tbl+15}, {ax+22} {tbl+37}, {ax+4} {tbl+37}" '
             f'stroke="{LEAK}" stroke-width="1.3" fill="none" marker-end="url(#arrowleak)"/>')
    o.append(f'<path d="M {ax} {tbl+15} C {ax+34} {tbl+15}, {ax+34} {tbl+59}, {ax+4} {tbl+59}" '
             f'stroke="{LEAK}" stroke-width="1.3" fill="none" marker-end="url(#arrowleak)"/>')

    note = tbl + 3 * th + 14
    o.append(text(L, note, "стрелки: ответ, данный на строке t, уже находится в контексте "
                           "строк t+1 и t+2", size=9, fill=LEAK))
    o.append(text(L, note + 10, "arrows: the answer given on row t is already in the context "
                                "of rows t+1 and t+2", size=8.5, fill=LEAK, style="italic"))
    o.append(text(L, note + 23,
                  f"измеренный AUC, {DISPLAY[EXAMPLE[0]]}, модель {EXAMPLE[1].upper()}, "
                  f"среднее по пяти прогонам", size=8.5, fill=MUTED, style="italic"))

    # --- stage 3: the two ways of reading it -------------------------------
    b = note + 47
    o.append(bilingual(L, b, "3. Две вероятности на уровне задания",
                       "3. Two question-level probabilities"))
    by = b + 26
    bw, bh = (R - L - 14) / 2, 52
    for x, ttl, en, val, col, bg in (
            (L, "усреднение трёх строк", "averaging the three rows", n["late"], LEAK, LEAK_BG),
            (L + bw + 14, "отдельный проход", "question-level inference",
             n["clean"], CLEAN, CLEAN_BG)):
        o.append(rect(x, by, bw, bh, fill=bg, stroke=col, sw=1.2))
        o.append(text(x + 12, by + 18, ttl, size=11.5, weight="600"))
        o.append(text(x + 12, by + 30, en, size=9, fill=MUTED, style="italic"))
        o.append(text(x + bw - 12, by + 42, f"AUC {val:.3f}", size=17, weight="700",
                      fill=col, anchor="end"))

    return "".join(o), int(by + bh + PAD)


# ------------------------------------------------------------------ figure 2

def figure2_svg(rows: list[dict]) -> tuple[str, int]:
    L, R = PAD, W - PAD
    left, right = L + 168, R - 56
    top = 60
    rh = 26

    lo, hi = 0.60, 0.92
    def sx(v: float) -> float:
        return left + (v - lo) / (hi - lo) * (right - left)

    o: list[str] = [defs()]
    o.append(bilingual(L, 12, "AUC при двух способах получения вероятности",
                       "AUC under the two ways of obtaining the probability"))

    # axis
    ay = top - 12
    for t in [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
        o.append(line(sx(t), ay + 4, sx(t), top + rh * len(rows) - 4,
                      stroke="#EDF0F3", sw=1))
        o.append(text(sx(t), ay, f"{t:.2f}", size=8.5, fill=MUTED, anchor="middle"))

    o.append(text(L, top - 13, "набор данных", size=8.5, fill=MUTED))
    o.append(text(L, top - 4, "dataset", size=8, fill=MUTED, style="italic"))
    o.append(text(left - 14, top - 13, "компонентов", size=8.5, fill=MUTED, anchor="end"))
    o.append(text(left - 14, top - 4, "на задание", size=8.5, fill=MUTED, anchor="end"))

    for i, r in enumerate(rows):
        cy = top + i * rh + rh / 2
        o.append(text(L, cy + 3.5, DISPLAY[r["dataset"]], size=10.5))
        o.append(text(left - 14, cy + 3.5, f"{r['kc_per_question']:.3f}", size=10,
                      fill=MUTED, anchor="end"))
        x1, x2 = sx(r["clean"]), sx(r["late"])
        if abs(x2 - x1) > 1.5:
            o.append(line(x1, cy, x2, cy, stroke=LEAK, sw=2.6))
            o.append(f'<circle cx="{x1}" cy="{cy}" r="4.2" fill="{CLEAN}"/>')
            o.append(f'<circle cx="{x2}" cy="{cy}" r="4.2" fill="{LEAK}"/>')
            pv = r.get("predicted")
            if pv == pv:
                o.append(f'<circle cx="{sx(pv):.2f}" cy="{cy}" r="6.4" fill="none" '
                         f'stroke="{INK}" stroke-width="1.4"/>')
            # Подпись отодвигается за самый правый маркер строки, включая
            # кружок предсказания, иначе он её задевает.
            right_edge = max(x1, x2, sx(pv) + 6.4 if pv == pv else x2)
            o.append(text(right_edge + 10, cy + 3.5, f"+{r['gap']:.4f}", size=10,
                          fill=LEAK, weight="600"))
        else:
            o.append(f'<circle cx="{x1}" cy="{cy}" r="5.6" fill="none" '
                     f'stroke="{CLEAN}" stroke-width="1.6"/>')
            o.append(f'<circle cx="{x1}" cy="{cy}" r="2.4" fill="{LEAK}"/>')
        o.append(line(L, top + i * rh, R, top + i * rh, stroke="#F0F2F5"))

    yb = top + rh * len(rows)
    o.append(line(L, yb, R, yb, stroke=RULE))

    # legend, and one note instead of repeating it on every coinciding row
    ly = yb + 19
    o.append(f'<circle cx="{L+5}" cy="{ly-3.5}" r="4.2" fill="{CLEAN}"/>')
    o.append(text(L + 16, ly, "отдельный проход на уровне задания  ·  question-level inference",
                  size=9.5, fill=INK))
    o.append(f'<circle cx="{L+5}" cy="{ly+12}" r="4.2" fill="{LEAK}"/>')
    o.append(text(L + 16, ly + 15.5, "усреднение строк задания  ·  averaging the rows of a question",
                  size=9.5, fill=INK))
    o.append(f'<circle cx="{L+5}" cy="{ly+27.5}" r="5.6" fill="none" '
             f'stroke="{CLEAN}" stroke-width="1.6"/>')
    o.append(f'<circle cx="{L+5}" cy="{ly+27.5}" r="2.4" fill="{LEAK}"/>')
    o.append(f'<circle cx="{L+5}" cy="{ly+43}" r="6.4" fill="none" stroke="{INK}" '
             f'stroke-width="1.4"/>')
    o.append(text(L + 18, ly + 46.5,
                  "предсказание механизма · prediction from the mechanism", size=9))
    o.append(text(L + 16, ly + 31, "обе точки совпадают  ·  the two dots coincide",
                  size=9.5, fill=INK))
    return "".join(o), int(ly + 46.5 + PAD + 6)


# ------------------------------------------------------------------ figure 3

MODEL_LABEL = {"dkt": "DKT", "sakt": "SAKT", "akt": "AKT", "simplekt": "simpleKT"}
MODEL_ORDER = ["dkt", "sakt", "akt", "simplekt"]


def figure3_rows() -> list[dict]:
    """Per dataset and model: AUC on the first row of a question and on continuations.

    Only the datasets where unrolling happens are drawn. On the other three the
    continuation class is empty, so a dumbbell there would have nothing to join; table 3
    states that in words, which is the right place for an absence.
    """
    rep = pd.read_csv(CROSS / "leakage_by_repeat_summary.csv")
    kcq = pd.read_csv(CROSS / "leakage_kc_per_question.csv")
    wide = rep.pivot_table(index=["dataset", "model"], columns="position_class",
                           values="auc_mean").reset_index()
    wide = wide.dropna(subset=["first_row", "repeat_row"])
    wide = wide.merge(kcq[["dataset", "kc_per_question"]], on="dataset")
    wide = wide[wide.kc_per_question > 1.0].sort_values("kc_per_question", ascending=False)

    out = []
    for ds in dict.fromkeys(wide.dataset):
        g = wide[wide.dataset == ds]
        models = [m for m in MODEL_ORDER if m in set(g.model)]
        out.append({
            "dataset": ds,
            "kc": float(g.kc_per_question.iloc[0]),
            "models": [{"model": m,
                        "first": float(g[g.model == m].first_row.iloc[0]),
                        "repeat": float(g[g.model == m].repeat_row.iloc[0])}
                       for m in models],
        })
    return out


def figure3_svg(groups: list[dict]) -> tuple[str, int]:
    L, R = PAD, W - PAD
    # Левое поле держит две колонки: название модели у фиксированной границы и
    # значение на первой строке вплотную к своей точке. 186 единиц — минимум, при
    # котором они не сходятся на наборе с самой левой точкой (EdNet, SAKT 0.626).
    left, right = L + 186, R - 46
    top, rh, gap = 58, 17, 16
    lo, hi = 0.60, 1.005

    def sx(v: float) -> float:
        return left + (v - lo) / (hi - lo) * (right - left)

    o: list[str] = [defs()]
    o.append(bilingual(L, 14, "AUC по позициям внутри задания",
                       "AUC by position inside a question", size=12))
    for v in (0.6, 0.7, 0.8, 0.9, 1.0):
        o.append(text(sx(v), top - 12, f"{v:.1f}", size=8.5, fill=MUTED, anchor="middle"))

    y = top
    for g in groups:
        o.append(line(L, y - 7, R, y - 7, stroke=RULE, sw=0.6))
        o.append(text(L, y + 5, DISPLAY[g["dataset"]], size=9.5, weight="600"))
        o.append(text(L, y + 15, f'{g["kc"]:.3f} комп./зад.', size=8, fill=MUTED))
        for v in (0.6, 0.7, 0.8, 0.9, 1.0):
            o.append(line(sx(v), y - 3, sx(v), y + len(g["models"]) * rh - 6,
                          stroke=RULE, sw=0.5, dash="2 3"))
        for k, m in enumerate(g["models"]):
            cy = y + k * rh + 6
            x1, x2 = sx(m["first"]), sx(m["repeat"])
            o.append(line(x1, cy, x2, cy, stroke=LEAK, sw=2.2))
            o.append(f'<circle cx="{x1:.2f}" cy="{cy:.2f}" r="3.6" fill="{CLEAN}" '
                     f'stroke="#FFFFFF" stroke-width="1.2"/>')
            o.append(f'<circle cx="{x2:.2f}" cy="{cy:.2f}" r="3.6" fill="{LEAK}" '
                     f'stroke="#FFFFFF" stroke-width="1.2"/>')
            o.append(text(left - 42, cy + 3, MODEL_LABEL[m["model"]], size=8.5,
                          anchor="end", fill=INK))
            o.append(text(x1 - 6, cy + 3, f'{m["first"]:.3f}', size=8,
                          fill=CLEAN, anchor="end"))
            o.append(text(x2 + 6, cy + 3, f'{m["repeat"]:.4f}', size=8, fill=LEAK))
        y += len(g["models"]) * rh + gap

    ly = y + 2
    o.append(f'<circle cx="{L+5}" cy="{ly-3.5}" r="3.6" fill="{CLEAN}"/>')
    o.append(text(L + 14, ly, "первая строка задания · first row of a question", size=8.5))
    o.append(f'<circle cx="{L+5}" cy="{ly+11}" r="3.6" fill="{LEAK}"/>')
    o.append(text(L + 14, ly + 14.5, "строки-продолжения · continuation rows", size=8.5))
    return "".join(o), int(ly + 26)


# ------------------------------------------------------------------- render

def wrap(body: str, height: int) -> str:
    return (f'<!doctype html><html><head><meta charset="utf-8">'
            f'<style>html,body{{margin:0;padding:0;background:#fff}}'
            f'svg{{display:block}}</style></head><body>'
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{height}" '
            f'viewBox="0 0 {W} {height}">{body}</svg></body></html>')


def render(html: str, stem: Path, height: int, do_render: bool) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    svg = html.split("<body>", 1)[1].rsplit("</body>", 1)[0]
    stem.with_suffix(".svg").write_text(
        f'<?xml version="1.0" encoding="UTF-8"?>\n{svg}', encoding="utf-8")
    if not do_render:
        return
    if CHROME is None:
        print("  Безголовый браузер не найден: собран только SVG. Укажите путь "
              "в переменной окружения CHROME_PATH или установите Chrome/Chromium.")
        return
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "f.html"
        src.write_text(html, encoding="utf-8")
        common = [str(CHROME), "--headless", "--disable-gpu", "--no-sandbox",
                  "--hide-scrollbars", "--virtual-time-budget=2000"]
        subprocess.run(common + [f"--screenshot={stem.with_suffix('.png')}",
                                 f"--window-size={W},{height}",
                                 f"--force-device-scale-factor={SCALE}", str(src)],
                       check=True, capture_output=True)
        subprocess.run(common + [f"--print-to-pdf={stem.with_suffix('.pdf')}",
                                 "--no-pdf-header-footer", str(src)],
                       check=True, capture_output=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=FIG_DIR)
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()

    n = figure1_numbers()
    body, h = figure1_svg(n)
    render(wrap(body, h), args.out_dir / "c1_fig1_mechanism", h, not args.no_render)
    print(f"рис. 1: {W}×{h} ед. -> {W*SCALE}×{h*SCALE} px, ширина "
          f"{W*SCALE/300*2.54:.1f} см при 300 dpi")
    print("   числа:", {k: round(v, 4) for k, v in n.items()})

    rows = figure2_rows()
    body, h = figure2_svg(rows)
    render(wrap(body, h), args.out_dir / "c1_fig2_inflation", h, not args.no_render)
    print(f"рис. 2: {W}×{h} ед. -> {W*SCALE}×{h*SCALE} px, ширина "
          f"{W*SCALE/300*2.54:.1f} см при 300 dpi")
    for r in rows:
        print(f"   {r['dataset']:20} комп/зад {r['kc_per_question']:.3f}  "
              f"чистый {r['clean']:.4f}  усреднение {r['late']:.4f}  разрыв {r['gap']:+.4f}")

    groups = figure3_rows()
    body, h = figure3_svg(groups)
    render(wrap(body, h), args.out_dir / "c1_fig3_by_position", h, not args.no_render)
    print(f"рис. 3: {W}×{h} ед. -> {W*SCALE}×{h*SCALE} px, ширина "
          f"{W*SCALE/300*2.54:.1f} см при 300 dpi")
    for g in groups:
        rng_f = [m["first"] for m in g["models"]]
        rng_r = [m["repeat"] for m in g["models"]]
        print(f"   {g['dataset']:20} первые {min(rng_f):.3f}-{max(rng_f):.3f}  "
              f"продолжения {min(rng_r):.4f}-{max(rng_r):.4f}")

    if W * SCALE / 300 * 2.54 > 14.0:
        print("ВНИМАНИЕ: шире 14 см — журнал не примет")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
