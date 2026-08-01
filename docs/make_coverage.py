"""Redraw the coverage graphic from a survey, instead of typing numbers into it.

    python conformance/survey_discs.py --predictors /path/to/archive > survey.json
    python docs/make_coverage.py survey.json

The figure in README.md says which (precision, predictor) combinations the
synthetic corpus sweeps, and which of them real files actually reach. Its cell
counts were written by hand and went stale twice without anyone noticing —
first at 61,921 frames, then at 81,172 — because a number inside an image is
not something a grep over the documents will ever find.

The numbers come out of the survey now, so the only way to change them is to
measure something different. Layout, palette and wording are left exactly as
drawn; each cell is rewritten in place rather than the grid being regenerated,
which keeps this honest about being a small edit to someone's drawing.

A real frame's predictor is read from its JPEG scan header, so this needs
`--predictors` output; a plain survey cannot know it.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

SVG = Path(__file__).with_name("coverage.svg")

# Geometry of the grid, read off the figure as drawn.
LEFT, TOP, STEP, SIZE = 108, 74, 37, 34
PRECISIONS = range(2, 17)
PREDICTORS = range(1, 8)

FILLED = '<rect x="{x}" y="{y}" width="34" height="34" rx="4" fill="#2a78d6"/>'
LABEL = ('<text x="{tx}" y="{ty}" font-size="9" text-anchor="middle" '
         'fill="#ffffff" font-weight="600" '
         'font-variant-numeric="tabular-nums">{text}</text>')
EMPTY = ('<rect x="{x}" y="{y}" width="34" height="34" rx="4" fill="#dfe6ee" '
         'stroke="#c6d2de" stroke-width="1"/>')


def short(count: int) -> str:
    """Cell labels are nine pixels tall and thirty-four wide; five digits do not fit."""
    if count >= 10_000:
        return f"{count / 1000:.0f}k"
    if count >= 1_000:
        return f"{count / 1000:.1f}k"
    return str(count)


def counts_from(survey: dict) -> dict[tuple[int, int], int]:
    """(precision, predictor) -> files, from the survey's combination tally."""
    if not survey.get("combinations"):
        raise SystemExit(
            "This survey has no combinations. Re-run survey_discs.py with "
            "--predictors: the predictor lives in the JPEG scan header, not "
            "in a DICOM tag, so a plain survey cannot know it.")

    tally: dict[tuple[int, int], int] = {}
    pattern = re.compile(r"P(\d+) pred(\d+)")
    for key, count in survey["combinations"]:
        match = pattern.match(key)
        if match:
            cell = (int(match.group(1)), int(match.group(2)))
            tally[cell] = tally.get(cell, 0) + count
    return tally


def rewrite(text: str, tally: dict[tuple[int, int], int]) -> tuple[str, int]:
    """Replace every grid cell in place, leaving the rest of the drawing alone."""
    changed = 0
    for predictor in PREDICTORS:
        y = TOP + (predictor - 1) * STEP
        for precision in PRECISIONS:
            x = LEFT + (precision - 2) * STEP
            # A cell is one rect, optionally followed by its count label.
            cell = re.compile(
                rf'<rect x="{x}" y="{y}" width="34" height="34"[^>]*/>'
                rf'(?:\s*<text x="{x + SIZE // 2}" y="{y + 20}"[^>]*>[^<]*</text>)?')
            count = tally.get((precision, predictor), 0)
            if count:
                new = (FILLED.format(x=x, y=y) + "\n"
                       + LABEL.format(tx=x + SIZE // 2, ty=y + 20,
                                      text=short(count)))
            else:
                new = EMPTY.format(x=x, y=y)

            text, hits = cell.subn(lambda _match, new=new: new, text, count=1)
            if not hits:
                raise SystemExit(
                    f"no cell found at precision {precision}, predictor "
                    f"{predictor}. The figure's geometry has moved and this "
                    "script's constants have not; fix them rather than letting "
                    "it write numbers into the wrong squares.")
            changed += hits
    return text, changed


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {argv[0]} <survey.json from --predictors>", file=sys.stderr)
        return 2

    survey = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    tally = counts_from(survey)
    text, changed = rewrite(SVG.read_text(encoding="utf-8"), tally)

    frames = survey["lossless_jpeg_frames"]
    text = re.sub(r"Also seen in real files — [^<]*",
                  f"Also seen in real files — {len(tally)} combinations, "
                  f"{frames:,} frames", text)
    SVG.write_text(text, encoding="utf-8")

    print(f"redrew {SVG.name}: {changed} cells, {len(tally)} of them reached, "
          f"{frames:,} frames")
    for (precision, predictor), count in sorted(tally.items()):
        print(f"  precision {precision:>2}  predictor {predictor}  {count:>8,}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
