"""
Builds a code snapshot (editor-style image) of every cleansing step,
used in the report: images/code/

    python src/make_snapshots.py
"""

import inspect
import io
import textwrap
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pygments import highlight
from pygments.formatters import ImageFormatter
from pygments.lexers import PythonLexer

sys.path.insert(0, str(Path(__file__).parent))
import data_cleansing as dc  # noqa: E402

IMG = dc.IMG_DIR
(IMG / "code").mkdir(parents=True, exist_ok=True)

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
    print(f"{len(steps)} code snapshots written to images/code/")


if __name__ == "__main__":
    main()
