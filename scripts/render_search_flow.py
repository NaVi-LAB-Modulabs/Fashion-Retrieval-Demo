"""Render the same diagram as SVG and PNG. Requires Pillow for regeneration."""

from html import escape
from math import hypot
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SCALE = 160
image = Image.new("RGB", (12 * SCALE, 15 * SCALE), "#f8f9f5")
draw = ImageDraw.Draw(image)
svg = [
    '<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="1500" viewBox="0 0 12 15" role="img" aria-labelledby="title desc">',
    '<title id="title">Fashion retrieval flow</title>',
    '<desc id="desc">Hard filters run before attribute top 10 and description top 10 (exact cosine by default). '
    'Their union is ranked by exact description similarity, required attributes and requested style axes.</desc>',
    '<rect width="12" height="15" fill="#f8f9f5"/>',
]


def text(x, y, value, *, fontsize=12, color="#4c5d56", bold=False, vertical=False):
    font_name = "segoeuib.ttf" if bold else "segoeui.ttf"
    fonts = [Path("C:/Windows/Fonts") / font_name,
             Path("/usr/share/fonts/truetype/dejavu") / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")]
    font_path = next((path for path in fonts if path.exists()), None)
    font = ImageFont.truetype(str(font_path), round(fontsize * SCALE / 72)) if font_path else ImageFont.load_default()
    spacing = fontsize / 72 * 1.55
    lines = value.splitlines()
    for index, line in enumerate(lines):
        center_y = 15 - y + (index - (len(lines) - 1) / 2) * spacing
        transform = f' transform="rotate(-90 {x} {center_y})"' if vertical else ""
        svg.append(f'<text x="{x}" y="{center_y}" text-anchor="middle" dominant-baseline="central" '
                   f'font-family="Segoe UI, DejaVu Sans, sans-serif" font-size="{fontsize / 72}" '
                   f'font-weight="{"700" if bold else "400"}" fill="{color}"{transform}>{escape(line)}</text>')
        if vertical:
            bounds = draw.textbbox((0, 0), line, font=font)
            label = Image.new("RGBA", (bounds[2] - bounds[0] + 12, bounds[3] - bounds[1] + 12))
            ImageDraw.Draw(label).text((6 - bounds[0], 6 - bounds[1]), line, font=font, fill=color)
            label = label.rotate(90, expand=True)
            image.paste(label, (round(x * SCALE - label.width / 2), round(center_y * SCALE - label.height / 2)), label)
        else:
            draw.text((x * SCALE, center_y * SCALE), line, font=font, fill=color, anchor="mm")


def box(x, y, width, height, title, detail="", *, fill="#ffffff", accent="#253b36"):
    svg.append(f'<rect x="{x}" y="{15-y-height}" width="{width}" height="{height}" '
               f'rx="0.16" fill="{fill}" stroke="#ccd7cd" stroke-width="0.017"/>')
    draw.rounded_rectangle((x * SCALE, (15-y-height) * SCALE, (x+width) * SCALE, (15-y) * SCALE),
                           radius=.16 * SCALE, fill=fill, outline="#ccd7cd", width=3)
    text(x + width / 2, y + height - .29, title, fontsize=15, bold=True, color=accent)
    if detail:
        text(x + width / 2, y + (height - .48) / 2, detail)


def arrow(points, *, dashed=False):
    screen_points = [(x, 15-y) for x, y in points]
    dash = ' stroke-dasharray="0.08 0.05"' if dashed else ""
    coordinates = " ".join(f"{x},{y}" for x, y in screen_points)
    svg.append(f'<polyline points="{coordinates}" fill="none" stroke="#73877b" stroke-width="0.02"{dash}/>')
    for start, end in zip(screen_points, screen_points[1:]):
        length = hypot(end[0]-start[0], end[1]-start[1])
        if dashed:
            offset = 0.0
            while offset < length:
                stop = min(length, offset + .08)
                a = tuple((start[i] + (end[i]-start[i]) * offset/length) * SCALE for i in range(2))
                b = tuple((start[i] + (end[i]-start[i]) * stop/length) * SCALE for i in range(2))
                draw.line([a, b], fill="#73877b", width=3)
                offset += .13
        else:
            draw.line([(start[0]*SCALE, start[1]*SCALE), (end[0]*SCALE, end[1]*SCALE)], fill="#73877b", width=3)
    start, end = screen_points[-2:]
    length = hypot(end[0]-start[0], end[1]-start[1])
    dx, dy = (end[0]-start[0])/length, (end[1]-start[1])/length
    triangle = [end, (end[0]-.12*dx+.06*dy, end[1]-.12*dy-.06*dx),
                (end[0]-.12*dx-.06*dy, end[1]-.12*dy+.06*dx)]
    svg.append('<polygon points="' + " ".join(f"{x},{y}" for x, y in triangle) + '" fill="#73877b"/>')
    draw.polygon([(x*SCALE, y*SCALE) for x, y in triangle], fill="#73877b")


text(6, 14.55, "Fashion retrieval flow", fontsize=25, bold=True, color="#203e32")
text(6, 14.13, "Hard filters → two candidate lists → weighted ranking", color="#63756b")
box(2, 12.95, 8, .75, "English search query")
arrow([(6, 12.95), (6, 12.65)])
box(2, 11.55, 8, 1.10, "Parse the query",
    "Category · explicit attributes · exclusions\nDescription query · requested style axes")
arrow([(6, 11.55), (6, 11.18)])
box(2, 9.88, 8, 1.30, "1  Apply hard filters",
    "Named category + all required positive attributes\nExcluded attributes remove matches; attribute threshold applies",
    fill="#edf4e9")
arrow([(6, 9.88), (6, 9.52)])
box(2, 8.62, 8, .90, "All items that pass the filters",
    "Both candidate lists use this population")
arrow([(6, 8.62), (6, 8.38), (3.45, 8.38), (3.45, 8.05)])
arrow([(6, 8.62), (6, 8.38), (8.55, 8.38), (8.55, 8.05)])
box(1.05, 6.25, 4.8, 1.80, "2a  Attribute top 10",
    "Combined score of required attributes\nconfidence × group metric\nOmit this branch if none are required",
    fill="#edf4e9")
box(6.15, 6.25, 4.8, 1.80, "2b  Description top 10",
    "Query embedding + stored descriptions\nExact cosine top 10 (optional HNSW)\nSearch all filtered, model-compatible IDs",
    fill="#edf2fa", accent="#2e4e75")
arrow([(3.45, 6.25), (3.45, 5.97), (6, 5.97), (6, 5.68)])
arrow([(8.55, 6.25), (8.55, 5.97), (6, 5.97), (6, 5.68)])
box(2, 4.70, 8, .98, "Merge IDs and remove duplicates",
    "At most 20 candidates")
arrow([(6, 4.70), (6, 4.32)])
box(2, 2.32, 8, 2.00, "3  Calculate final scores",
    "Description: exact cosine similarity — 55%\nAttributes: same required attributes — 35%\nStyle: only axes mentioned in the query — 10%",
    fill="#eef0e6")
arrow([(2, 12.10), (.48, 12.10), (.48, 3.20), (2, 3.20)], dashed=True)
text(.25, 6.75, "Requested style targets → final scoring only",
     vertical=True, fontsize=10, color="#63756b")
arrow([(6, 2.32), (6, 1.95)])
box(2, 1.03, 8, .92, "Sort by final score → results",
    "Apply minimum final score and requested result limit")
text(6, .65, "Default weights are adjustable and normalized over available components.",
     fontsize=10, color="#63756b")
text(6, .32, "No compatible embeddings: attribute candidates only. No eligible items: empty results.",
     fontsize=10, color="#63756b")

(ROOT / "web" / "search-flow.svg").write_text("\n".join([*svg, "</svg>"]) + "\n", encoding="utf-8")
image.save(ROOT / "web" / "search-flow.png")
print("web/search-flow.svg\nweb/search-flow.png")
