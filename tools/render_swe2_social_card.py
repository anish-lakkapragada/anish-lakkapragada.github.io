"""Render a social cover in the style of the article's v16-B film.

Run: uv run --with pillow --with numpy python tools/render_swe2_social_card.py
Optionally pass the film directory as the first argument to locate its fonts.
The recorded paths and endpoint headings are data, not decorative curves.
"""

import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets/swe-2-extended"
FILM = (Path(sys.argv[1]) if len(sys.argv) > 1 else
        ROOT.parent / "swe2-extended/videos/full-film/v16B-v15edit")
FONT = FILM / "assets/SpaceGrotesk.ttf"
MATH_FONT = FILM / "src/node_modules/katex/dist/fonts/KaTeX_Math-Italic.ttf"
MAIN_FONT = FILM / "src/node_modules/katex/dist/fonts/KaTeX_Main-Regular.ttf"
SCALE = 3
W, H = 1200, 630
BG = "#fdfdfd"
INK = "#1d2b2e"
COLORS = [INK, "#ef7758", "#d3a13f", "#008c77", "#739aac", "#8981bc"]


def rgb(color, opacity=1):
    return (*tuple(int(color[i:i + 2], 16) for i in (1, 3, 5)), round(255 * opacity))


def font(size, weight=500, path=FONT):
    face = ImageFont.truetype(str(path), round(size * SCALE))
    if path == FONT:
        face.set_variation_by_axes([weight])
    return face


def smooth_path(points):
    """Use the film's centered average, preserving both recorded endpoints."""
    smoothed = np.array([
        points[max(0, i - 5):min(len(points), i + 6)].mean(axis=0)
        for i in range(len(points))
    ])
    correction = np.linspace(points[0] - smoothed[0], points[-1] - smoothed[-1], len(points))
    return smoothed + correction


def main():
    data = json.loads((ASSETS / "data/video-figures.json").read_text())
    assert data["methods"] == ["fixed", "adapt_0", "adapt_0.25", "adapt_0.5", "adapt_0.75", "adapt_1"]
    paths = np.array(data["medium_threads"])
    canvas = Image.new("RGB", (W * SCALE, H * SCALE), BG)
    draw = ImageDraw.Draw(canvas, "RGBA")

    def line(points, color, width=1, opacity=1):
        draw.line([(round(x * SCALE), round(y * SCALE)) for x, y in points],
                  fill=rgb(color, opacity), width=max(1, round(width * SCALE)))

    def dot(x, y, radius, color, opacity=1):
        draw.ellipse(tuple(v * SCALE for v in (x - radius, y - radius, x + radius, y + radius)),
                     fill=rgb(color, opacity))

    def text(value, x, y, size, color=INK, weight=500, anchor="lt", path=FONT):
        draw.text((round(x * SCALE), round(y * SCALE)), value,
                  font=font(size, weight, path), fill=color, anchor=anchor)

    # A large, open compass: the same six methods and colors as the film.
    ox, oy, radius = 918, 474, 278
    units = radius / 101
    line([(ox - radius - 10, oy), (1160, oy)], INK, .65, .19)
    line([(ox, oy + 20), (ox, oy - radius - 32)], INK, .65, .19)
    arc = [(ox - math.cos(t) * radius, oy - math.sin(t) * radius)
           for t in np.linspace(0, math.pi, 240)]
    line(arc, INK, .75, .2)
    for angle in range(0, 181, 15):
        t = math.radians(angle)
        line([(ox - math.cos(t) * radius, oy - math.sin(t) * radius),
              (ox - math.cos(t) * (radius + 5), oy - math.sin(t) * (radius + 5))], INK, .7, .24)

    for method, color in enumerate(COLORS):
        for raw in paths[method]:
            points = smooth_path(raw)
            xy = points * np.array([units, -units]) + np.array([ox, oy])
            for a, b in zip(xy[:-1], xy[1:]):
                midpoint = (a + b) / 2
                # Like the film, fade the crowded common origin. Fade the
                # outer crop as well so long fixed-penalty paths stay quiet.
                distance = np.linalg.norm(midpoint - [ox, oy])
                fade = min(1, distance / 66) ** 1.5
                crop = min(1, max(0, (midpoint[0] - 560) / 45),
                           max(0, (1180 - midpoint[0]) / 65),
                           max(0, (midpoint[1] - 80) / 65),
                           max(0, (548 - midpoint[1]) / 28))
                opacity = (.065 if method == 0 else .31) * fade * crop
                if opacity > .004:
                    line([a, b], color, .65, opacity)

    for j, alpha in enumerate([0, .25, .5, .75, 1], start=1):
        angle = math.atan2(alpha, 1 - alpha)
        for distance in np.arange(12, radius - 8, 8):
            line([(ox - math.cos(angle) * distance, oy - math.sin(angle) * distance),
                  (ox - math.cos(angle) * (distance + 2), oy - math.sin(angle) * (distance + 2))],
                 COLORS[j], .85, .38)
        lx = ox - math.cos(angle) * (radius + 39)
        ly = oy - math.sin(angle) * (radius + 39)
        if alpha == 1:
            lx -= 15
            ly -= 4
        # Typeset alpha in the same math face used by the film.
        suffix = f" = {alpha:g}"
        alpha_font, value_font = font(19, path=MATH_FONT), font(19, path=MAIN_FONT)
        aw = draw.textlength("α", font=alpha_font)
        total = aw + draw.textlength(suffix, font=value_font)
        x = lx * SCALE - total / 2
        draw.text((x, ly * SCALE), "α", font=alpha_font, fill=COLORS[j], anchor="lm")
        draw.text((x + aw, ly * SCALE), suffix, font=value_font, fill=COLORS[j], anchor="lm")

    # Place each measured endpoint's heading on the compass. Only radius is
    # jittered, deterministically, to separate nearby samples (as in the film).
    for method, color in enumerate(COLORS):
        rng = np.random.default_rng(500 + method)
        for endpoint in paths[method, :, -1, :]:
            angle = math.atan2(endpoint[1], -endpoint[0])
            r = radius - 8 - rng.uniform(0, 12)
            dot(ox - math.cos(angle) * r, oy - math.sin(angle) * r,
                1.75, color, .36 if method == 0 else .85)
    dot(ox, oy, 6.5, BG)
    dot(ox, oy, 3.8, INK)

    # Generous type, matching the film's Space Grotesk and evergreen accent.
    text("Steering", 60, 167, 78)
    text("the Pareto", 60, 251, 78)
    text("frontier.", 60, 335, 78, "#008c77")

    output = ASSETS / "social-preview.jpg"
    canvas = canvas.resize((2400, 1260), Image.Resampling.LANCZOS)
    canvas.save(output, quality=94, subsampling=0, optimize=True, progressive=True)
    print(f"Rendered {output} ({output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
