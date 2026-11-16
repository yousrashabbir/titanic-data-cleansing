"""
Builds the images used in the report:
  * code snapshots of every cleansing step (images/code/)
  * console-output snapshots of every step (images/output/)
  * before / after views of the dataset rows that had problems (images/)

Run from the project root after data_cleansing.py:
    python src/make_snapshots.py
"""

import contextlib
import inspect
import io
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from pygments import highlight
from pygments.formatters import ImageFormatter
from pygments.lexers import PythonLexer

sys.path.insert(0, str(Path(__file__).parent))
import data_cleansing as dc  # noqa: E402

IMG = dc.IMG_DIR
(IMG / "code").mkdir(parents=True, exist_ok=True)
(IMG / "output").mkdir(parents=True, exist_ok=True)

MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
BG, FG, BAR = "#1e1f22", "#e6e6e6", "#2b2d31"


def add_window_frame(img, title):
    """Wrap an image in an editor-style window with a title bar."""
    pad, bar_h = 18, 34
    w, h = img.size
    out = Image.new("RGB", (w + 2 * pad, h + bar_h + pad), BG)
    d = ImageDraw.Draw(out)
    d.rectangle([0, 0, out.width, bar_h], fill=BAR)
    for i, c in enumerate(["#ff5f57", "#febc2e", "#28c840"]):
        d.ellipse([14 + i * 20, 11, 26 + i * 20, 23], fill=c)
    d.text((86, 9), title, fill="#a9abb0", font=ImageFont.truetype(MONO, 14))
    out.paste(img, (pad, bar_h + pad // 2))
    return out


def code_snapshot(func, name):
    src = textwrap.dedent(inspect.getsource(func))
    png = highlight(src, PythonLexer(), ImageFormatter(
        style="one-dark", font_name="DejaVu Sans Mono", font_size=15,
        line_numbers=True, line_number_bg=BG, line_number_fg="#5c6370",
        image_pad=6, line_pad=4))
    img = Image.open(io.BytesIO(png)).convert("RGB")
    add_window_frame(img, "data_cleansing.py").save(IMG / "code" / f"{name}.png")


def output_snapshot(text, name, max_lines=40):
    lines = text.rstrip("\n").splitlines()
    if len(lines) > max_lines:
        lines = lines[:max_lines] + [f"... ({len(lines) - max_lines} more lines)"]
    font = ImageFont.truetype(MONO, 15)
    lh = 21
    width = max(font.getlength(l) for l in lines + ["$ python src/data_cleansing.py"])
    img = Image.new("RGB", (int(width) + 20, lh * (len(lines) + 1) + 12), BG)
    d = ImageDraw.Draw(img)
    d.text((6, 4), "$ python src/data_cleansing.py", fill="#28c840", font=font)
    for i, line in enumerate(lines, start=1):
        d.text((6, 4 + i * lh), line, fill=FG, font=font)
    add_window_frame(img, "Terminal - output").save(IMG / "output" / f"{name}.png")


def run(func, *args):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = func(*args)
    return result, buf.getvalue()


def table_image(df, path, title, highlight_mask=None, hl_color="#fbd3c3"):
    """Render a DataFrame as a spreadsheet-style table image."""
    shown = df.astype(object).where(df.notnull(), "NaN")
    n_rows, n_cols = shown.shape
    col_w = [max(len(str(c)), *(len(str(v)) for v in shown[c])) for c in shown]
    fig_w = max(sum(col_w) * 0.105 + 0.6, 6)
    fig, ax = plt.subplots(figsize=(fig_w, 0.3 * (n_rows + 1) + 0.3))
    ax.axis("off")
    tbl = ax.table(cellText=shown.values, colLabels=shown.columns,
                   cellLoc="center", loc="center",
                   colWidths=[w / sum(col_w) for w in col_w])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1, 1.45)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#c9c8c3")
        if r == 0:
            cell.set_facecolor("#1f4e79")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("white" if r % 2 else "#f4f6f9")
            if highlight_mask is not None and highlight_mask.iat[r - 1, c]:
                cell.set_facecolor(hl_color)
                cell.set_text_props(weight="bold")
    ax.set_title(title, loc="left", fontsize=11, pad=6)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    steps = [
        ("01_load", dc.load_data), ("02_inspect", dc.inspect_data),
        ("03_duplicates", dc.check_duplicates),
        ("04_missing_report", dc.missing_value_report),
        ("05_clean_text", dc.clean_text), ("06_title", dc.extract_title),
        ("07_embarked", dc.impute_embarked), ("08_age", dc.impute_age),
        ("09_cabin", dc.handle_cabin), ("10_zero_fare", dc.fix_zero_fares),
        ("11_dtypes", dc.convert_dtypes), ("12_outliers", dc.detect_outliers),
        ("13_features", dc.engineer_features),
        ("14_validate", dc.validate_and_save),
    ]
    for name, func in steps:
        code_snapshot(func, name)

    # Run the pipeline step by step, capturing each step's console output
    raw, out = run(dc.load_data); output_snapshot(out, "01_load")
    _, out = run(dc.inspect_data, raw[["PassengerId", "Survived", "Pclass",
                                       "Sex", "Age", "SibSp", "Parch",
                                       "Fare", "Embarked"]])
    dtypes_txt = "Data types:\n" + raw.dtypes.to_string()
    output_snapshot(dtypes_txt, "02_inspect_dtypes")
    output_snapshot("Summary statistics (numeric columns):\n"
                    + raw.describe().round(2).to_string(), "02_inspect_describe")
    df, out = run(dc.check_duplicates, raw); output_snapshot(out, "03_duplicates")
    _, out = run(dc.missing_value_report, df, "before")
    output_snapshot(out, "04_missing_before")
    df, out = run(dc.clean_text, df); output_snapshot(out, "05_clean_text")
    df, out = run(dc.extract_title, df); output_snapshot(out, "06_title")
    df, out = run(dc.impute_embarked, df); output_snapshot(out, "07_embarked")
    df, out = run(dc.impute_age, df); output_snapshot(out, "08_age")
    df, out = run(dc.handle_cabin, df); output_snapshot(out, "09_cabin")
    df, out = run(dc.fix_zero_fares, df); output_snapshot(out, "10_zero_fare")
    df, out = run(dc.convert_dtypes, df); output_snapshot(out, "11_dtypes")
    _, out = run(dc.detect_outliers, df); output_snapshot(out, "12_outliers")
    df, out = run(dc.engineer_features, df); output_snapshot(out, "13_features")
    _, out = run(dc.missing_value_report, df, "after")
    output_snapshot(out, "04_missing_after")
    df, out = run(dc.validate_and_save, df); output_snapshot(out, "14_validate")

    # ---- Before / after views of the dataset --------------------------------
    clean = pd.read_csv(dc.CLEAN_PATH)
    cols = ["PassengerId", "Name", "Pclass", "Sex", "Age", "Fare",
            "Cabin", "Embarked"]

    def short(frame):
        f = frame[cols].copy()
        f["Name"] = f["Name"].str.slice(0, 26)
        return f.reset_index(drop=True)

    # Rows that had missing Age / Cabin / Embarked
    ids = [6, 18, 20, 29, 62, 830, 33, 37]
    b = short(raw[raw.PassengerId.isin(ids)])
    a = short(clean[clean.PassengerId.isin(ids)])
    mask = raw[raw.PassengerId.isin(ids)][cols].isnull().reset_index(drop=True)
    table_image(b, IMG / "rows_missing_before.png",
                "BEFORE cleaning - rows with missing values (orange = NaN)",
                mask, "#fbd3c3")
    table_image(a, IMG / "rows_missing_after.png",
                "AFTER cleaning - the same rows, missing values filled "
                "(green = filled)", mask, "#c8ecd9")

    # Rows with hidden missing values (Fare = 0)
    zid = raw.loc[raw.Fare == 0, "PassengerId"].head(6).tolist()
    b = short(raw[raw.PassengerId.isin(zid)])
    a = short(clean[clean.PassengerId.isin(zid)])
    zmask = pd.DataFrame(False, index=b.index, columns=cols)
    zmask["Fare"] = True
    table_image(b, IMG / "zero_fare_before.png",
                "BEFORE - Fare recorded as 0.00 (hidden missing value)",
                zmask, "#fbd3c3")
    table_image(a, IMG / "zero_fare_after.png",
                "AFTER - Fare replaced with the class median", zmask, "#c8ecd9")

    # Whitespace problems
    wid = [16, 858]
    b = raw[raw.PassengerId.isin(wid)][["PassengerId", "Name"]].copy()
    b["Name"] = b["Name"].map(lambda s: f"'{s}'")
    a = clean[clean.PassengerId.isin(wid)][["PassengerId", "Name"]].copy()
    a["Name"] = a["Name"].map(lambda s: f"'{s}'")
    table_image(b.reset_index(drop=True), IMG / "whitespace_before.png",
                "BEFORE - trailing space inside the quotes")
    table_image(a.reset_index(drop=True), IMG / "whitespace_after.png",
                "AFTER - whitespace removed")

    # First rows of the raw and cleaned files
    table_image(short(raw.head(8)), IMG / "raw_head.png",
                "Raw dataset - first 8 rows (orange = NaN)",
                raw.head(8)[cols].isnull().reset_index(drop=True))
    new_cols = ["PassengerId", "Title", "Age", "Fare", "Cabin", "Deck",
                "HasCabin", "Embarked", "FamilySize", "IsAlone", "AgeGroup"]
    table_image(clean.head(8)[new_cols], IMG / "clean_head.png",
                "Cleaned dataset - first 8 rows (new columns included)")
    print("Report images written to", IMG)


if __name__ == "__main__":
    main()
