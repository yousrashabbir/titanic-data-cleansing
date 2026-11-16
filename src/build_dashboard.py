"""
Builds the complete visual dashboard as one image (pure Python).

    python src/build_dashboard.py
      -> images/dashboard/<panel>.png          one image per panel
      -> dashboard/titanic_cleansing_dashboard.png   full dashboard
      -> dashboard/titanic_cleansing_dashboard.pdf   same, as PDF
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from dashboard_data import compute  # noqa: E402
import visuals as vz  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DPI = 220

# Panel order in the two dashboard columns (filled shortest-column-first)
ORDER = ["overview", "pipeline", "missing_before", "matrix", "profile",
         "duplicates", "age", "rows_before", "rows_after", "embarked", "cabin",
         "text", "titles", "fare", "dtypes", "outliers", "features",
         "missing_after", "validation", "preview"]


def header(width_px, D):
    """Dark title band at the top of the dashboard."""
    o = D["overview"]
    w_in = width_px / DPI
    fig = plt.figure(figsize=(w_in, 1.55), facecolor=vz.NAVY)
    fig.text(0.024, 0.66, "Titanic Dataset · Data Cleansing Dashboard",
             fontsize=26, weight="bold", color="white", va="center")
    fig.text(0.024, 0.33, "Python · pandas · Matplotlib   |   Kaggle Titanic train.csv   |   "
             f"{o['rows_before']} passengers",
             fontsize=11.5, color="#c7d2e6", va="center")
    for i, (lab, val, col) in enumerate([
            ("MISSING CELLS", f"{o['missing_before']} → {o['missing_after']}", "#7ee2b8"),
            ("DUPLICATES", "0 found", "#7ee2b8"),
            ("CHECKS", f"{sum(v['ok'] for v in D['validation'])}/{len(D['validation'])} passed", "#7ee2b8")]):
        x = 0.69 + i * 0.105
        fig.text(x, 0.68, lab, fontsize=9, color="#9fb0cc", weight="bold", va="center")
        fig.text(x, 0.38, val, fontsize=17, color=col, weight="bold", va="center")
    path = ROOT / "images" / "dashboard" / "_header.png"
    fig.savefig(path, dpi=DPI, facecolor=vz.NAVY)
    plt.close(fig)
    return Image.open(path).convert("RGB")


def build():
    D, _, _ = compute()
    paths = vz.save_panels(D, ROOT / "images" / "dashboard", dpi=DPI)
    panels = [Image.open(paths[n]).convert("RGB") for n in ORDER]

    pw = panels[0].width
    gap = 24
    width = 2 * pw + 3 * gap
    cols = [[], []]
    heights = [0, 0]
    for im in panels:
        c = 0 if heights[0] <= heights[1] else 1
        cols[c].append(im)
        heights[c] += im.height + gap

    head = header(width, D)
    bg = tuple(int(vz.BG[i:i + 2], 16) for i in (1, 3, 5))
    canvas = Image.new("RGB", (width, head.height + max(heights) + gap), bg)
    canvas.paste(head, (0, 0))
    for c, col in enumerate(cols):
        y = head.height + gap
        x = gap + c * (pw + gap)
        for im in col:
            canvas.paste(im, (x, y))
            y += im.height + gap

    out_dir = ROOT / "dashboard"
    out_dir.mkdir(exist_ok=True)
    canvas.save(out_dir / "titanic_cleansing_dashboard.png", optimize=True)
    canvas.save(out_dir / "titanic_cleansing_dashboard.pdf", resolution=DPI)
    (ROOT / "images" / "dashboard" / "_header.png").unlink()
    # Top section of the dashboard, used as a preview in the report / README
    top = canvas.crop((0, 0, width, min(canvas.height, int(width * 1.12))))
    top.resize((1650, int(top.height * 1650 / width))).save(ROOT / "images" / "dashboard_top.png")
    print(f"Dashboard saved: dashboard/titanic_cleansing_dashboard.png "
          f"({canvas.width} x {canvas.height} px) and .pdf")


if __name__ == "__main__":
    build()
