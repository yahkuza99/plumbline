"""The figure generator has to put each number in the right square.

`docs/make_coverage.py` exists because the counts in the README's figure were
typed in by hand and went stale twice — a number inside an SVG survives every
grep anyone runs over the documents. Replacing that with a script only helps if
the script is right, and its bad outcome is not a crash: it is a figure that
looks correct and says the wrong thing about which parts of the format have
real-file evidence behind them.

So these check placement, not just that it runs.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
SCRIPT = ROOT / "docs" / "make_coverage.py"
SVG = ROOT / "docs" / "coverage.svg"

sys.path.insert(0, str(ROOT / "docs"))
import make_coverage                                        # noqa: E402


def _survey(combinations, frames=1000):
    return {"lossless_jpeg_frames": frames,
            "combinations": [[key, count] for key, count in combinations]}


@pytest.fixture
def original():
    """Restore the committed figure whatever a test does to it."""
    before = SVG.read_text(encoding="utf-8")
    yield before
    SVG.write_text(before, encoding="utf-8")


def test_each_count_lands_in_its_own_square(original):
    """precision 12, predictor 3 must colour the cell at precision 12, predictor 3."""
    survey = _survey([("P12 pred3 comp1 pt0 restart-rows:none", 4242)])
    text, _ = make_coverage.rewrite(original, make_coverage.counts_from(survey))

    x = make_coverage.LEFT + (12 - 2) * make_coverage.STEP
    y = make_coverage.TOP + (3 - 1) * make_coverage.STEP
    assert f'<text x="{x + 17}" y="{y + 20}"' in text
    assert text.count("4.2k") == 1

    # And no other square is filled in. The failure that matters here is a
    # figure claiming real-file evidence for combinations that have none.
    #
    # Counted at grid coordinates rather than by colour: the legend below the
    # grid uses the same blue for its key swatch, and a test that counted every
    # rect of that colour would have been off by one for a reason that has
    # nothing to do with what it is checking.
    filled = re.findall(r'<rect x="(\d+)" y="(\d+)" width="34" height="34"[^>]*'
                        r'fill="#2a78d6"', text)
    assert len(filled) == 1, f"filled squares at {filled}"


def test_a_survey_without_predictors_is_refused(original):
    """A plain survey cannot know the predictor, so it must not be guessed at."""
    with pytest.raises(SystemExit, match="--predictors"):
        make_coverage.counts_from({"lossless_jpeg_frames": 1, "combinations": []})


def test_moved_geometry_stops_rather_than_mislabels(original):
    """If the drawing changes under it, refusing beats writing into wrong squares."""
    damaged = original.replace(f'<rect x="{make_coverage.LEFT}" '
                               f'y="{make_coverage.TOP}"', '<rect x="999" y="999"', 1)
    with pytest.raises(SystemExit, match="no cell found"):
        make_coverage.rewrite(damaged, {(2, 1): 5})


@pytest.mark.parametrize("count, shown", [
    (7, "7"), (999, "999"), (1_000, "1.0k"), (2_651, "2.7k"),
    (9_999, "10.0k"), (10_000, "10k"), (68_369, "68k"),
])
def test_labels_fit_the_cell(count, shown):
    """Thirty-four pixels wide at nine point: five digits do not fit."""
    assert make_coverage.short(count) == shown
    assert len(shown) <= 5


def test_it_runs_end_to_end_and_leaves_valid_svg(original, tmp_path):
    """The whole path, as CI or a maintainer would run it."""
    import xml.etree.ElementTree as ElementTree

    survey = tmp_path / "survey.json"
    survey.write_text(json.dumps(_survey([
        ("P8 pred1 comp1 pt0 restart-rows:1", 3099),
        ("P16 pred1 comp1 pt0 restart-rows:none", 28497),
    ], frames=31_596)), encoding="utf-8")

    result = subprocess.run([sys.executable, str(SCRIPT), str(survey)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    text = SVG.read_text(encoding="utf-8")
    ElementTree.fromstring(text)                            # still parses
    assert "Also seen in real files — 2 combinations, 31,596 frames" in text
    # The seven row labels are outside the grid and must survive a redraw.
    assert len(re.findall(r">predictor \d<", text)) == 7
