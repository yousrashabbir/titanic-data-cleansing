"""
Dashboard visuals for the Titanic data-cleansing project (pure Python).

Every function returns a Matplotlib Figure styled as a dashboard card.
The same figures are used three ways:
  * saved as PNG panels for the report            (python src/visuals.py)
  * combined into one dashboard image              (dashboard/ folder)
  * shown in the interactive Streamlit dashboard   (streamlit run app.py)
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Wedge
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# ----------------------------------------------------------------------------
# Design tokens
# ----------------------------------------------------------------------------
BG = "#eef1f6"        # page background around the card
CARD = "#ffffff"
INK = "#111827"       # primary text
MUTED = "#6b7280"     # secondary text
LINE = "#e5e7eb"      # borders / grid
TRACK = "#f1f3f6"     # empty progress track
NAVY = "#1f2a44"
BLUE, BLUE_L = "#2a78d6", "#e6f0fb"
GREEN, GREEN_L = "#14995f", "#def5ea"
ORANGE, ORANGE_L = "#eb6834", "#fde8df"
VIOLET, MAGENTA, YELLOW, GREY = "#6250d6", "#e87ba4", "#eda100", "#8a8f98"

plt.rcParams.update({
    "font.family": "Inter", "font.size": 9, "axes.edgecolor": LINE,
    "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.titlesize": 10, "axes.titleweight": "semibold",
    "axes.titlecolor": INK, "figure.dpi": 100,
})

W = 7.5   # panel width in inches


# ----------------------------------------------------------------------------
# Layout helpers (all positions in inches from the bottom-left corner)
# ----------------------------------------------------------------------------
class Panel:
    """A dashboard card: rounded white box, title, subtitle and step badge."""

    def __init__(self, h, title, subtitle="", step=None, w=W):
        self.w, self.h = w, h
        self.fig = plt.figure(figsize=(w, h), facecolor=BG)
        self.c = self.fig.add_axes([0, 0, 1, 1], zorder=-1)
        self.c.set_xlim(0, w); self.c.set_ylim(0, h); self.c.axis("off")
        m = 0.08
        self.c.add_patch(FancyBboxPatch((m, m), w - 2 * m, h - 2 * m,
                         boxstyle="round,pad=0,rounding_size=0.16",
                         fc=CARD, ec=LINE, lw=1))
        x = 0.38
        if step is not None:
            pill(self.c, x, h - 0.48, f"STEP {step}", BLUE_L, BLUE, size=7.5)
            x += 0.78
        self.text(x, h - 0.48, title, 12.5, INK, "bold", va="center")
        if subtitle:
            self.text(0.38, h - 0.78, subtitle, 8.6, MUTED, va="center")

    def text(self, x, y, s, size=9, color=INK, weight="normal", ha="left",
             va="center", **kw):
        return self.c.text(x, y, s, fontsize=size, color=color, weight=weight,
                           ha=ha, va=va, **kw)

    def width(self, t):
        """Width of a text object in inches."""
        r = self.fig.canvas.get_renderer()
        return t.get_window_extent(r).width / self.fig.dpi

    def box(self, x, y, w, h, fc, ec="none", r=0.1, lw=1, z=1):
        self.c.add_patch(FancyBboxPatch((x, y), w, h,
                         boxstyle=f"round,pad=0,rounding_size={r}",
                         fc=fc, ec=ec, lw=lw, zorder=z))

    def axes(self, x, y, w, h):
        ax = self.fig.add_axes([x / self.w, y / self.h, w / self.w, h / self.h])
        style(ax)
        return ax


def pill(c, x, y, s, fc, tc, size=8, weight="bold", ha="left"):
    """Rounded label ('chip'). Returns its width in inches."""
    wdt = 0.0062 * size * len(s) + 0.22
    x0 = x if ha == "left" else (x - wdt / 2 if ha == "center" else x - wdt)
    hgt = 0.028 * size
    c.add_patch(FancyBboxPatch((x0, y - hgt / 2), wdt, hgt,
                boxstyle=f"round,pad=0,rounding_size={hgt / 2}", fc=fc, ec="none", zorder=3))
    c.text(x0 + wdt / 2, y, s, fontsize=size, color=tc, weight=weight,
           ha="center", va="center", zorder=4)
    return wdt


def check_icon(c, x, y, r=0.11, ok=True):
    col = GREEN if ok else ORANGE
    c.add_patch(Circle((x, y), r, fc=col, ec="none", zorder=3))
    if ok:
        c.plot([x - r * .45, x - r * .1, x + r * .5], [y, y - r * .38, y + r * .38],
               color="white", lw=1.6, solid_capstyle="round", zorder=4)
    else:
        c.text(x, y, "!", color="white", ha="center", va="center", fontsize=8,
               weight="bold", zorder=4)


def style(ax):
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(LINE)
    ax.tick_params(length=0, labelsize=8)
    ax.grid(axis="y", color=LINE, lw=0.7)
    ax.set_axisbelow(True)
    ax.set_facecolor(CARD)


def kpi(p, x, y, w, h, label, before, after, note="", good=True):
    """KPI tile: label, 'before -> after' values and a note."""
    p.box(x, y, w, h, "#fafbfc", LINE, r=0.1)
    p.text(x + 0.16, y + h - 0.24, label.upper(), 7.2, MUTED, "bold")
    if before is None:
        p.text(x + 0.16, y + h / 2 - 0.05, str(after), 20, GREEN if good else INK, "bold")
    else:
        t = p.text(x + 0.16, y + h / 2 - 0.05, str(before), 15, ORANGE, "bold")
        x2 = x + 0.16 + p.width(t) + 0.08
        a = p.text(x2, y + h / 2 - 0.05, "→", 13, MUTED)
        p.text(x2 + p.width(a) + 0.08, y + h / 2 - 0.05, str(after),
               20, GREEN if good else INK, "bold")
    if note:
        p.text(x + 0.16, y + 0.2, note, 7.4, MUTED)


def table(p, x_top, y_top, cols, rows, widths, rh=0.27, size=7.8, fills=None,
          bold_cols=(), aligns=None):
    """Draw a clean UI table. fills[(r, c)] = (fill, text colour)."""
    x0 = x_top
    total = sum(widths)
    p.box(x0, y_top - rh, total, rh, NAVY, r=0.06)
    xs = np.cumsum([0] + widths[:-1]) + x0
    for c_i, (cx, col) in enumerate(zip(xs, cols)):
        p.text(cx + 0.08, y_top - rh / 2, col, size, "white", "bold")
    for r_i, row in enumerate(rows):
        y = y_top - rh * (r_i + 2)
        if r_i % 2:
            p.c.add_patch(plt.Rectangle((x0, y), total, rh, fc="#f8f9fb", ec="none"))
        for c_i, (cx, val) in enumerate(zip(xs, row)):
            fc_tc = (fills or {}).get((r_i, c_i))
            tc = INK
            if fc_tc:
                p.box(cx + 0.03, y + 0.035, widths[c_i] - 0.06, rh - 0.07, fc_tc[0], r=0.05)
                tc = fc_tc[1]
            p.text(cx + 0.08, y + rh / 2, str(val), size, tc,
                   "bold" if (c_i in bold_cols or fc_tc) else "normal")
        p.c.plot([x0, x0 + total], [y, y], color=LINE, lw=0.6)


def donut(ax, values, colors, center_big, center_small):
    ax.pie(values, colors=colors, startangle=90, counterclock=False,
           wedgeprops=dict(width=0.32, edgecolor="white", linewidth=2))
    ax.text(0, 0.08, center_big, ha="center", va="center", fontsize=15,
            weight="bold", color=INK)
    ax.text(0, -0.22, center_small, ha="center", va="center", fontsize=7.5, color=MUTED)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.spines["bottom"].set_visible(False)


def legend_dot(p, x, y, color, label):
    p.c.add_patch(Circle((x, y), 0.055, fc=color, ec="none"))
    p.text(x + 0.11, y, label, 7.8, MUTED)


# ----------------------------------------------------------------------------
# Panels
# ----------------------------------------------------------------------------
def overview_panel(D):
    o = D["overview"]
    p = Panel(3.75, "Data Quality Overview",
              "Titanic passenger dataset (Kaggle, train.csv) - raw file vs cleaned file")
    tw, th, gap = 2.15, 1.18, 0.12
    tiles = [
        ("Missing cells", o["missing_before"], o["missing_after"], "Age, Cabin, Embarked filled"),
        ("Completeness", f'{o["completeness_before"]}%', "100%", "share of non-empty cells"),
        ("Duplicate rows", None, "0", "no duplicate values found"),
        ("Hidden zero fares", o["zero_fares"], 0, "Fare = 0.00 replaced"),
        ("Columns", o["cols_before"], o["cols_after"], "7 derived columns added"),
        ("Rows kept", o["rows_before"], o["rows_after"], "no passenger removed"),
    ]
    for i, (lab, b, a, note) in enumerate(tiles):
        r, c = divmod(i, 3)
        x = 0.38 + c * (tw + gap)
        y = 3.75 - 1.1 - (r + 1) * th - r * gap + 0.02
        kpi(p, x, y, tw, th, lab, b, a, note)
    return p.fig


def pipeline_panel():
    steps = ["Load", "Inspect", "Duplicates", "Missing\nreport", "Clean\ntext",
             "Titles", "Embarked", "Age", "Cabin", "Zero\nfares", "Data\ntypes",
             "Outliers", "Derived\ncolumns", "Validate\n& save"]
    p = Panel(3.45, "Cleansing Pipeline", "14 steps, each one a Python function in src/data_cleansing.py")
    colors = {0: BLUE, 1: BLUE, 2: GREEN, 3: ORANGE, 4: VIOLET, 5: VIOLET, 6: ORANGE,
              7: ORANGE, 8: ORANGE, 9: ORANGE, 10: VIOLET, 11: YELLOW, 12: VIOLET, 13: GREEN}
    for row in range(2):
        y = 2.3 - row * 1.12
        xs = [0.75 + i * 0.99 for i in range(7)]
        p.c.plot([xs[0], xs[-1]], [y, y], color=LINE, lw=2, zorder=1)
        for i, x in enumerate(xs):
            k = row * 7 + i
            p.c.add_patch(Circle((x, y), 0.2, fc=colors[k], ec="white", lw=2, zorder=3))
            p.text(x, y, str(k + 1), 8.5, "white", "bold", ha="center", zorder=4)
            p.text(x, y - 0.34, steps[k], 7.6, INK, ha="center", va="top", linespacing=1.1)
    lx = 0.4
    for col, lab in [(BLUE, "Explore"), (GREEN, "Check"), (ORANGE, "Fix missing values"),
                     (VIOLET, "Standardise / create"), (YELLOW, "Outliers")]:
        legend_dot(p, lx, 0.3, col, lab)
        lx += 0.32 + 0.068 * len(lab)
    return p.fig


def profile_panel(D):
    prof = D["profile"]
    rh = 0.29
    h = 1.25 + rh * (len(prof) + 1) + 0.2
    p = Panel(h, "Column Profile - Raw Dataset", "891 rows x 12 columns, as loaded from the CSV file", step="1-2")
    xs = dict(col=0.38, dtype=1.55, bar=2.5, pct=4.2, miss=4.85, uniq=5.6, info=6.25)
    y0 = h - 1.15
    for k, lab in [("col", "COLUMN"), ("dtype", "TYPE"), ("bar", "COMPLETENESS"),
                   ("miss", "MISSING"), ("uniq", "UNIQUE"), ("info", "RANGE / TOP")]:
        p.text(xs[k], y0, lab, 7, MUTED, "bold")
    p.c.plot([0.38, W - 0.38], [y0 - 0.16, y0 - 0.16], color=LINE, lw=0.8)
    for i, r in enumerate(prof):
        y = y0 - rh * (i + 1) - 0.05
        p.text(xs["col"], y, r["col"], 8.2, INK, "semibold")
        tcol = {"int64": (BLUE_L, BLUE), "float64": ("#efeafd", VIOLET)}.get(r["dtype"], ("#f1f2f4", "#4b5563"))
        pill(p.c, xs["dtype"], y, r["dtype"], tcol[0], tcol[1], size=7)
        frac = r["non_null"] / 891
        p.box(xs["bar"], y - 0.06, 1.55, 0.12, TRACK, r=0.06)
        p.box(xs["bar"], y - 0.06, max(1.55 * frac, 0.12), 0.12,
              GREEN if frac == 1 else (ORANGE if frac < 0.5 else YELLOW), r=0.06)
        p.text(xs["pct"], y, f"{100 * frac:.1f}%", 7.8, INK)
        if r["missing"]:
            pill(p.c, xs["miss"], y, str(r["missing"]), ORANGE_L, ORANGE, size=7.5)
        else:
            p.text(xs["miss"] + 0.05, y, "0", 7.8, MUTED)
        p.text(xs["uniq"], y, str(r["unique"]), 7.8, INK)
        info = (f'{r["min"]:g} – {r["max"]:g}' if "min" in r else r["top"])
        p.text(xs["info"], y, info[:14], 7.6, MUTED)
    return p.fig


def duplicates_panel(D):
    d = D["duplicates"]
    p = Panel(3.3, "Duplicate Check", "Four independent checks for repeated records", step=3)
    checks = [("Identical rows", d["full_rows"], "df.duplicated()"),
              ("Repeated PassengerId", d["passenger_id"], "unique key"),
              ("Same row, other ID", d["ignoring_id"], "ID ignored"),
              ("Same name + ticket", d["name_ticket"], "same person twice")]
    tw = 1.58
    for i, (lab, n, note) in enumerate(checks):
        x = 0.38 + i * (tw + 0.105)
        p.box(x, 1.3, tw, 1.12, "#fafbfc", LINE, r=0.1)
        p.text(x + 0.15, 2.18, lab, 7.8, MUTED, "semibold")
        p.text(x + 0.15, 1.78, str(n), 22, INK, "bold")
        check_icon(p.c, x + tw - 0.26, 1.8, ok=(n == 0))
        p.text(x + 0.15, 1.45, note, 7, MUTED)
    p.box(0.38, 0.34, W - 0.76, 0.72, GREEN_L, r=0.1)
    check_icon(p.c, 0.72, 0.7, r=0.16)
    p.text(1.0, 0.8, "No duplicate values found", 11.5, GREEN, "bold")
    p.text(1.0, 0.53, "All 891 rows are unique - no records were removed.", 8.3, "#14532d")
    return p.fig


def missing_panel(D, state="before"):
    data = D["missing_before"] if state == "before" else D["missing_after"]
    rh = 0.27
    h = 1.55 + rh * len(data)
    total = sum(r["n"] for r in data)
    title = "Missing Values - " + ("Before Cleaning" if state == "before" else "After Cleaning")
    sub = (f"{total} empty cells in 3 of 12 columns" if state == "before"
           else f"{len(data)} columns (12 original + 7 derived) - {total} missing cells")
    p = Panel(h, title, sub, step=4 if state == "before" else None)
    y0 = h - 1.15
    p.text(0.38, y0, "COLUMN", 7, MUTED, "bold")
    p.text(1.95, y0, "MISSING ROWS  (track = 891 passengers)", 7, MUTED, "bold")
    p.text(6.15, y0, "COUNT", 7, MUTED, "bold")
    tw = 3.95
    for i, r in enumerate(data):
        y = y0 - rh * (i + 1) - 0.04
        p.text(0.38, y, r["col"], 8.1, INK, "semibold" if r["n"] else "normal")
        p.box(1.95, y - 0.065, tw, 0.13, TRACK, r=0.065)
        if r["n"]:
            p.box(1.95, y - 0.065, max(tw * r["n"] / 891, 0.13), 0.13, ORANGE, r=0.065)
            pill(p.c, 6.15, y, f'{r["n"]}  ·  {r["pct"]}%', ORANGE_L, ORANGE, size=7.2)
        else:
            check_icon(p.c, 6.25, y, r=0.075)
            p.text(6.4, y, "0", 7.8, GREEN, "bold")
    return p.fig


def matrix_panel(D):
    p = Panel(4.6, "Missing-Value Map", "Each column is a variable, each thin line a passenger - orange = empty cell")
    for k, (key, label, x, w) in enumerate([("matrix_before", "BEFORE", 0.38, 2.9),
                                             ("matrix_after", "AFTER", 3.75, 3.37)]):
        m = D[key]
        arr = np.array([[int(ch) for ch in s] for s in m.values()]).T
        ax = p.axes(x, 0.95, w, 2.4)
        ax.imshow(arr, aspect="auto", interpolation="nearest",
                  cmap=LinearSegmentedColormap.from_list("m", ["#e9edf3", ORANGE]), vmin=0, vmax=1)
        ax.set_xticks(range(len(m)))
        ax.set_xticklabels(list(m.keys()), rotation=90, fontsize=6.3)
        ax.set_yticks([])
        ax.grid(False)
        ax.spines["bottom"].set_visible(False)
        n = int(arr.sum())
        pill(p.c, x, 3.55, f"{label}  ·  {n} missing", ORANGE_L if n else GREEN_L,
             ORANGE if n else GREEN, size=7.5)
    return p.fig


def text_panel(D):
    p = Panel(4.1, "Text Cleaning", "Stray spaces removed and categories standardised", step=5)
    y = 2.75
    for r in D["whitespace"]:
        p.text(0.38, y + 0.02, f'ID {r["id"]}', 7.6, MUTED, "bold")
        p.box(1.0, y - 0.17, 2.7, 0.38, "#fafbfc", LINE, r=0.07)
        t = p.text(1.1, y + 0.02, r["before"].rstrip(), 7.4, INK, family="DejaVu Sans Mono")
        L = len(r["before"].rstrip()) * 0.0618
        p.box(1.1 + L + 0.02, y - 0.08, 0.13, 0.2, ORANGE, r=0.03)
        p.text(3.88, y + 0.02, "→", 12, MUTED, ha="center")
        p.box(4.05, y - 0.17, 3.07, 0.38, GREEN_L, r=0.07)
        p.text(4.15, y + 0.02, r["after"], 7.4, "#14532d", family="DejaVu Sans Mono")
        y -= 0.55
    legend_dot(p, 1.05, 3.1, ORANGE, "trailing space found")
    legend_dot(p, 4.1, 3.1, GREEN, "trimmed with .str.strip()")
    p.c.plot([0.38, W - 0.38], [1.45, 1.45], color=LINE, lw=0.8)
    p.text(0.38, 1.12, "CATEGORIES AFTER CLEANING", 7, MUTED, "bold")
    x = 0.38
    p.text(x, 0.72, "Sex", 8.3, INK, "semibold"); x += 0.42
    for v in ["female", "male"]:
        x += pill(p.c, x, 0.72, v, BLUE_L, BLUE, size=7.6) + 0.08
    x += 0.3
    p.text(x, 0.72, "Embarked", 8.3, INK, "semibold"); x += 0.85
    for v in ["C", "Q", "S"]:
        x += pill(p.c, x, 0.72, v, BLUE_L, BLUE, size=7.6) + 0.08
    p.text(x + 0.3, 0.72, "one spelling per category", 7.8, MUTED)
    p.text(0.38, 0.38, "Columns checked: Name (2 fixed), Sex, Ticket, Cabin, Embarked (0 issues)", 7.6, MUTED)
    return p.fig


TITLE_COL = {"Mr": BLUE, "Miss": MAGENTA, "Mrs": VIOLET, "Master": YELLOW, "Rare": GREY}


def titles_panel(D):
    p = Panel(4.6, "Passenger Titles", "17 raw titles extracted from Name, grouped into 5 clean categories", step=6)
    raw = D["titles_raw"]
    ax = p.axes(1.35, 0.4, 2.6, 3.0)
    names = [r["title"] for r in raw][::-1]
    vals = [r["n"] for r in raw][::-1]
    cols = [TITLE_COL[r["group"]] for r in raw][::-1]
    ax.barh(names, vals, color=cols, height=0.7)
    for i, v in enumerate(vals):
        ax.text(v + 8, i, str(v), va="center", fontsize=6.8, color=INK)
    ax.set_xlim(0, 600); ax.grid(axis="x", color=LINE, lw=0.6); ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=7.2, labelcolor=INK)
    ax.set_title("Raw titles", loc="left", fontsize=8.5)
    ax2 = p.axes(4.75, 0.4, 2.3, 2.65)
    g = D["titles_group"]
    ax2.bar([r["title"] for r in g], [r["n"] for r in g],
            color=[TITLE_COL[r["title"]] for r in g], width=0.62)
    for i, r in enumerate(g):
        ax2.text(i, r["n"] + 10, str(r["n"]), ha="center", fontsize=8, weight="bold", color=INK)
    ax2.set_ylim(0, 580); ax2.tick_params(axis="x", labelsize=8, labelcolor=INK)
    ax2.set_title("After grouping", loc="left", fontsize=8.5)
    p.text(4.75, 3.52, "Mlle, Ms → Miss   Mme → Mrs", 7.4, MUTED)
    p.text(4.75, 3.32, "10 rare titles → Rare", 7.4, MUTED)
    return p.fig


def embarked_panel(D):
    e = D["embarked"]
    p = Panel(3.85, "Port of Embarkation", "2 missing ports (PassengerId 62 and 830) filled with the mode", step=7)
    for k, (state, x, title) in enumerate([("before", 0.5, "Before"), ("after", 4.0, "After")]):
        rows = e[state]
        ax = p.axes(x + 0.95, 0.42, 2.25, 2.15)
        labels = [r["port"] for r in rows][::-1]
        vals = [r["n"] for r in rows][::-1]
        cols = [ORANGE if r["code"] == "NaN" else (GREEN if (state == "after" and r["code"] == "S") else BLUE)
                for r in rows][::-1]
        ax.barh(labels, vals, color=cols, height=0.62)
        for i, v in enumerate(vals):
            ax.text(v + 12, i, f"{v}", va="center", fontsize=7.8, color=INK, weight="bold")
        ax.set_xlim(0, 780); ax.grid(axis="x", color=LINE, lw=0.6); ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", labelsize=7.8, labelcolor=INK)
        pill(p.c, x - 0.12, 2.75, title.upper(), ORANGE_L if state == "before" else GREEN_L,
             ORANGE if state == "before" else GREEN, size=7.4)
    p.text(6.0, 2.75, "S: 644 → 646 (+2)", 7.6, GREEN, "bold", ha="center")
    return p.fig


def age_panel(D):
    a = D["age"]
    p = Panel(6.15, "Age - Missing Values Filled", "177 missing ages (19.9%) filled with the median of the same Title and Pclass", step=8)
    ax = p.axes(0.75, 3.0, 4.35, 2.0)
    x = np.arange(len(a["bins"]))
    ax.bar(x, a["recorded"], color=BLUE, width=0.78, label="Recorded age")
    ax.bar(x, a["imputed"], bottom=a["recorded"], color=GREEN, width=0.78, label="Filled (imputed)")
    ax.set_xticks(x); ax.set_xticklabels(a["bins"], rotation=45, fontsize=6.5)
    ax.set_ylabel("Passengers", fontsize=7.5)
    ax.legend(frameon=False, fontsize=7.4, loc="upper right")
    ax.set_title("Age distribution after cleaning", loc="left", fontsize=8.5)
    tiles = [("Filled", f'{a["missing"]}', "missing → 0"),
             ("Median age", f'{a["median_before"]:g} → {a["median_after"]:g}', "years"),
             ("Mean age", f'{a["mean_before"]} → {a["mean_after"]}', "years")]
    for i, (lab, val, note) in enumerate(tiles):
        y = 4.55 - i * 0.78
        p.box(5.45, y, 1.67, 0.66, "#fafbfc", LINE, r=0.08)
        p.text(5.58, y + 0.5, lab.upper(), 6.8, MUTED, "bold")
        p.text(5.58, y + 0.24, val, 11.5, GREEN if i == 0 else INK, "bold")
        p.text(7.0, y + 0.5, note, 6.5, MUTED, ha="right")
    # median heat table
    p.text(0.38, 2.38, "MEDIAN AGE USED FOR EACH TITLE x CLASS  (small number = ages filled in that group)", 7, MUTED, "bold")
    m = D["age_medians"]; fc = D["age_filled_counts"]
    cmap = LinearSegmentedColormap.from_list("b", ["#eaf2fc", "#2a78d6"])
    cw, ch = 1.25, 0.34
    for j, cl in enumerate(["1st class", "2nd class", "3rd class"]):
        p.text(1.55 + j * (cw + 0.08) + cw / 2, 2.1, cl, 7.6, MUTED, "semibold", ha="center")
    for i, t in enumerate(m["titles"]):
        y = 1.7 - i * (ch + 0.05)
        p.text(0.38, y + ch / 2, t, 8, INK, "semibold")
        for j, v in enumerate(m["values"][i]):
            x0 = 1.55 + j * (cw + 0.08)
            if v is None:
                p.box(x0, y, cw, ch, "#f6f7f9", r=0.06)
                p.text(x0 + cw / 2, y + ch / 2, "–", 8, MUTED, ha="center")
                continue
            col = cmap(min(v / 50, 1))
            p.box(x0, y, cw, ch, col, r=0.06)
            tc = "white" if v > 28 else INK
            p.text(x0 + 0.13, y + ch / 2, f"{v:g} yrs", 8, tc, "bold")
            n = fc.get(t, {}).get(str(j + 1), 0)
            if n:
                p.text(x0 + cw - 0.1, y + ch / 2, f"{n} filled", 6.4, tc, ha="right")
    return p.fig


def cabin_panel(D):
    c = D["cabin"]
    p = Panel(3.55, "Cabin - 77% Missing", "Too sparse to estimate: labelled 'Unknown', with two new columns built from it", step=9)
    ax = p.axes(0.45, 0.4, 2.2, 2.2)
    donut(ax, [c["has"], c["unknown"]], [BLUE, "#d6dbe3"], "77.1%", "cabin unknown")
    legend_dot(p, 2.85, 2.2, BLUE, f'Cabin recorded  {c["has"]}')
    legend_dot(p, 2.85, 1.95, "#d6dbe3", f'Unknown  {c["unknown"]}')
    p.text(2.8, 1.5, "NEW COLUMNS", 7, MUTED, "bold")
    x = 2.8
    for v in ["HasCabin (0/1)", "Deck (A-G, T, U)"]:
        x += pill(p.c, x, 1.2, v, GREEN_L, GREEN, size=7.2) + 0.08
    ax2 = p.axes(5.25, 0.45, 1.95, 2.0)
    dk = c["decks"]
    ax2.bar([d["deck"] for d in dk], [d["n"] for d in dk], color=BLUE, width=0.66)
    for i, d in enumerate(dk):
        ax2.text(i, d["n"] + 1.5, str(d["n"]), ha="center", fontsize=6.8, color=INK)
    ax2.set_title("Known cabins by deck", loc="left", fontsize=8.2)
    ax2.set_ylim(0, 68)
    return p.fig


def rows_panel(D, state="before"):
    R = D["rows"]; cols = R["cols"]
    data = R[state]; before = R["before"]
    widths = [0.82, 1.98, 0.55, 0.63, 0.55, 0.7, 0.75, 0.76]
    h = 1.4 + 0.27 * (len(data) + 1)
    if state == "before":
        p = Panel(h, "Dataset Rows - Before Cleaning", "8 passengers taken straight from the raw file - orange = missing (NaN)")
    else:
        p = Panel(h, "Dataset Rows - After Cleaning", "The same 8 passengers in the cleaned file - green = value that was filled")
    rows, fills = [], {}
    for i, r in enumerate(data):
        row = []
        for j, cname in enumerate(cols):
            v = r[cname]
            was_missing = before[i][cname] is None
            if v is None:
                row.append("NaN"); fills[(i, j)] = (ORANGE_L, ORANGE)
            else:
                row.append(f"{v:g}" if isinstance(v, float) else v)
                if state == "after" and was_missing:
                    fills[(i, j)] = (GREEN_L, GREEN)
        rows.append(row)
    table(p, 0.38, h - 1.05, cols, rows, widths, fills=fills)
    return p.fig


def fare_panel(D):
    f = D["fare_zero"]
    p = Panel(5.2, "Hidden Missing Values - Fare = 0", "15 tickets recorded at 0.00 treated as missing and replaced with the class median", step=10)
    for i, cl in enumerate(["1", "2", "3"]):
        x = 0.38 + i * 2.27
        p.box(x, 3.3, 2.15, 0.82, "#fafbfc", LINE, r=0.08)
        p.text(x + 0.15, 3.92, f"CLASS {cl}", 7, MUTED, "bold")
        p.text(x + 0.15, 3.58, f'{f["by_class"][cl]} zero fares', 10.5, ORANGE, "bold")
        p.text(x + 2.0, 3.58, f'→ £{f["medians"][cl]}', 10.5, GREEN, "bold", ha="right")
    rows = f["rows"]
    half = (len(rows) + 1) // 2
    for k, chunk in enumerate([rows[:half], rows[half:]]):
        x0 = 0.38 + k * 3.45
        p.text(x0, 3.0, "ID", 7, MUTED, "bold"); p.text(x0 + 0.42, 3.0, "PASSENGER", 7, MUTED, "bold")
        p.text(x0 + 2.05, 3.0, "FARE", 7, MUTED, "bold")
        for i, r in enumerate(chunk):
            y = 2.7 - i * 0.3
            p.text(x0, y, str(r["id"]), 7.6, MUTED)
            p.text(x0 + 0.42, y, r["name"][:20], 7.6, INK)
            pill(p.c, x0 + 2.05, y, "0.00", ORANGE_L, ORANGE, size=7)
            p.text(x0 + 2.62, y, "→", 9, MUTED)
            pill(p.c, x0 + 2.8, y, f'{r["after"]:.2f}', GREEN_L, GREEN, size=7)
    return p.fig


DTYPE_COL = {"int64": (BLUE_L, BLUE), "float64": ("#efeafd", VIOLET), "int8": (BLUE_L, BLUE),
             "category": (GREEN_L, GREEN), "str": ("#f1f2f4", "#4b5563"), "-": ("#ffffff", "#c0c4cc")}


def dtypes_panel(D):
    d = D["dtypes"]
    half = (len(d) + 1) // 2
    h = 1.45 + 0.3 * half
    p = Panel(h, "Data Types Corrected", "Each column stored with the type that matches its meaning", step=11)
    for k, chunk in enumerate([d[:half], d[half:]]):
        x0 = 0.38 + k * 3.5
        p.text(x0, h - 1.12, "COLUMN", 7, MUTED, "bold")
        p.text(x0 + 1.35, h - 1.12, "BEFORE", 7, MUTED, "bold")
        p.text(x0 + 2.35, h - 1.12, "AFTER", 7, MUTED, "bold")
        for i, r in enumerate(chunk):
            y = h - 1.45 - i * 0.3
            p.text(x0, y, r["col"], 7.8, INK, "semibold")
            b = r["before"]
            pill(p.c, x0 + 1.35, y, "new" if r["new"] else b,
                 *(("#fff4d6", "#9a6700") if r["new"] else DTYPE_COL.get(b, DTYPE_COL["str"])), size=6.8)
            p.text(x0 + 2.18, y, "→", 8.5, MUTED, ha="center")
            pill(p.c, x0 + 2.35, y, r["after"], *DTYPE_COL.get(r["after"], DTYPE_COL["str"]), size=6.8)
    return p.fig


def outliers_panel(D):
    p = Panel(5.75, "Outlier Detection (IQR Rule)", "Values outside Q1 - 1.5 IQR / Q3 + 1.5 IQR flagged, checked and kept", step=12)
    for i, o in enumerate(D["outliers"]):
        x = 0.38 + i * 1.7
        p.box(x, 3.68, 1.6, 0.98, "#fafbfc", LINE, r=0.08)
        p.text(x + 0.14, 4.47, o["col"].upper(), 7, MUTED, "bold")
        p.text(x + 0.14, 4.14, f'{o["n"]}', 17, YELLOW if o["n"] else GREEN, "bold")
        p.text(x + 0.14 + 0.13 * len(str(o["n"])) + 0.07, 4.1, "flagged", 7.4, MUTED)
        p.text(x + 0.14, 3.84, f'bounds {o["low"]:g} to {o["high"]:g}', 6.6, MUTED)
    ax = p.axes(1.3, 1.18, 5.75, 2.1)
    fb = D["fare_box"]
    for i, b in enumerate(fb):
        y = 3 - i
        ax.add_patch(plt.Rectangle((b["q1"], y - 0.22), b["q3"] - b["q1"], 0.44,
                     fc=BLUE_L, ec=BLUE, lw=1.1))
        ax.plot([b["med"]] * 2, [y - 0.22, y + 0.22], color=NAVY, lw=1.8)
        ax.plot([b["lo"], b["q1"]], [y, y], color=BLUE, lw=1)
        ax.plot([b["q3"], b["hi"]], [y, y], color=BLUE, lw=1)
        for wv in (b["lo"], b["hi"]):
            ax.plot([wv, wv], [y - 0.1, y + 0.1], color=BLUE, lw=1)
        ax.scatter(b["points"], [y] * len(b["points"]), s=22, color=YELLOW,
                   edgecolor="white", linewidth=0.8, zorder=3)
    ax.set_yticks([3, 2, 1])
    ax.set_yticklabels([f'{n} class\n{b["n_out"]} outliers' for n, b in zip(["1st", "2nd", "3rd"], fb)],
                       fontsize=7.6, color=INK)
    ax.set_ylim(0.5, 3.5); ax.set_xlim(-5, 535)
    ax.grid(axis="x", color=LINE, lw=0.6); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Fare (£)", fontsize=7.5)
    ax.set_title("Fare by passenger class", loc="left", fontsize=8.5)
    p.box(0.38, 0.3, W - 0.76, 0.5, GREEN_L, r=0.08)
    check_icon(p.c, 0.66, 0.55, r=0.13)
    p.text(0.9, 0.55, "Decision: kept. They are real values (luxury 1st-class suites, large families), not entry errors.", 7.9, "#14532d", "semibold")
    return p.fig


def features_panel(D):
    f = D["features"]
    p = Panel(3.85, "Derived Columns", "FamilySize, IsAlone and AgeGroup built from the cleaned data", step=13)
    ax = p.axes(0.65, 0.5, 2.3, 2.15)
    ax.bar([r["size"] for r in f["family"]], [r["n"] for r in f["family"]], color=BLUE, width=0.7)
    ax.set_title("FamilySize", loc="left", fontsize=8.5); ax.set_xticks(range(1, 12))
    ax2 = p.axes(3.05, 0.45, 1.5, 2.0)
    donut(ax2, [f["alone"], f["with_family"]], [VIOLET, "#d6dbe3"],
          f'{100 * f["alone"] / 891:.0f}%', "travel alone")
    ax2.set_title("IsAlone", loc="left", fontsize=8.5)
    ax3 = p.axes(5.6, 0.5, 1.55, 2.15)
    g = f["agegroup"]
    ax3.barh([r["g"] for r in g][::-1], [r["n"] for r in g][::-1], color=GREEN, height=0.62)
    ax3.set_title("AgeGroup", loc="left", fontsize=8.5)
    ax3.grid(axis="x", color=LINE, lw=0.6); ax3.grid(axis="y", visible=False)
    ax3.tick_params(axis="y", labelsize=7.2, labelcolor=INK)
    return p.fig


def validation_panel(D):
    v = D["validation"]
    h = 1.3 + 0.36 * len(v)
    p = Panel(h, "Final Validation", "Automatic assert checks run before the clean file is saved", step=14)
    for i, r in enumerate(v):
        y = h - 1.2 - i * 0.36
        check_icon(p.c, 0.52, y, r=0.1, ok=r["ok"])
        p.text(0.75, y, r["check"], 8.5, INK, "semibold")
        p.text(3.3, y, r["detail"], 8, MUTED)
        if i < len(v) - 1:
            p.c.plot([0.38, 4.95], [y - 0.18, y - 0.18], color=LINE, lw=0.6)
    p.box(5.25, 0.4, 1.87, h - 1.35, GREEN_L, r=0.12)
    n_ok = sum(r["ok"] for r in v)
    p.text(6.19, h / 2 + 0.05, f"{n_ok}/{len(v)}", 26, GREEN, "bold", ha="center")
    p.text(6.19, h / 2 - 0.42, "CHECKS PASSED", 7.5, GREEN, "bold", ha="center")
    p.text(6.19, 0.62, "titanic_cleaned.csv saved", 6.8, "#14532d", ha="center")
    return p.fig


def preview_panel(D):
    pv = D["preview"]
    cols = pv["cols"]
    rows = [[(f"{r[c]:g}" if isinstance(r[c], float) else r[c]) for c in cols] for r in pv["rows"][:10]]
    widths = [0.52, 0.48, 0.45, 0.6, 0.56, 0.42, 0.5, 0.42, 0.62, 0.56, 0.48, 0.83]
    h = 1.4 + 0.25 * (len(rows) + 1)
    p = Panel(h, "Cleaned Dataset - Preview", "First 10 rows of titanic_cleaned.csv (selected columns)")
    labels = ["ID", "Surv.", "Class", "Title", "Sex", "Age", "Fare", "Deck", "Embarked", "Family", "Alone", "AgeGroup"]
    table(p, 0.38, h - 1.05, labels, rows, widths, rh=0.25, size=7)
    return p.fig


# ----------------------------------------------------------------------------
# Export
# ----------------------------------------------------------------------------
def all_panels(D):
    """Return {name: figure} for every dashboard panel, in report order."""
    return {
        "overview": overview_panel(D), "pipeline": pipeline_panel(),
        "profile": profile_panel(D), "duplicates": duplicates_panel(D),
        "missing_before": missing_panel(D, "before"), "matrix": matrix_panel(D),
        "text": text_panel(D), "titles": titles_panel(D),
        "embarked": embarked_panel(D), "age": age_panel(D), "cabin": cabin_panel(D),
        "rows_before": rows_panel(D, "before"), "rows_after": rows_panel(D, "after"),
        "fare": fare_panel(D), "dtypes": dtypes_panel(D), "outliers": outliers_panel(D),
        "features": features_panel(D), "missing_after": missing_panel(D, "after"),
        "validation": validation_panel(D), "preview": preview_panel(D),
    }


def save_panels(D, out_dir, dpi=220):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, fig in all_panels(D).items():
        path = out_dir / f"{name}.png"
        fig.savefig(path, dpi=dpi, facecolor=BG)
        plt.close(fig)
        paths[name] = path
    return paths


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from dashboard_data import compute
    D, _, _ = compute()
    root = Path(__file__).resolve().parents[1]
    paths = save_panels(D, root / "images" / "dashboard")
    print(f"{len(paths)} dashboard panels saved to images/dashboard/")
