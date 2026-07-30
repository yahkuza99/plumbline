"""Decode frames from scanners nobody here owns.

Everything else in this suite is either synthetic — written by our own encoder
from our own reading of the specification — or from discs collected in one
country. Both share a blind spot: they only contain what we thought to put in
them, or what the machines around here happened to write.

These eight frames come from the NCI Imaging Data Commons, from public
research collections gathered in the United States, under CC BY. They are
committed because they are small, redistributable, and cover two things
nothing else here does: **PET**, whose sample distribution looks nothing like
CT or MR, and scanner firmware from a different market.

They are bare JPEG frames, not DICOM files. Everything identifying a patient
lives in the DICOM tags around the pixel data, and none of that is here — the
frame is the entropy-coded image and nothing else. That is also what Plumbline
actually consumes, so testing the frame tests the thing under test.

Attribution and DOIs are in `tests/data/idc/manifest.json` and in NOTICE, as
CC BY requires.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import plumbline
from plumbline import reference

DATA = Path(__file__).parent / "data" / "idc"
CASES = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
IDS = [f"{c['manufacturer']}-{c['model']}-{c['modality']}".replace(" ", "-") for c in CASES]


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_it_decodes_to_the_recorded_pixels(case):
    """The pixels must be what they were when the frame was added.

    The hash is of the decoded samples, so it catches a change of value and a
    change of dtype alike — which is what a regression in this decoder would
    look like from the outside.
    """
    frame = (DATA / case["file"]).read_bytes()
    pixels = plumbline.decode(frame)

    assert list(pixels.shape) == case["shape"]
    assert str(pixels.dtype) == case["dtype"]
    assert hashlib.sha256(np.ascontiguousarray(pixels).tobytes()).hexdigest() \
        == case["sha256_pixels"], (
            f"{case['manufacturer']} {case['model']} decodes differently than when "
            "it was recorded — see CHANGELOG before changing this hash")


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_every_engine_agrees(case):
    """Whatever engine this machine picked must match the pure-Python one."""
    frame = (DATA / case["file"]).read_bytes()
    assert np.array_equal(plumbline.decode(frame), reference.decode(frame))


def test_no_patient_data_came_along():
    """A frame carrying DICOM structure would mean the wrong thing was saved.

    Cheap to check and worth checking on every run: the failure this guards
    against is not a decoding bug but a mistake in how the corpus was built,
    and that kind of mistake is silent.
    """
    for case in CASES:
        raw = (DATA / case["file"]).read_bytes()
        assert raw[:2] == b"\xff\xd8", f"{case['file']} does not start with SOI"
        assert b"DICM" not in raw[:1024], f"{case['file']} looks like a DICOM file"


def test_the_collection_is_redistributable():
    """Only CC BY. Anything non-commercial would quietly poison the licence.

    Plumbline is Apache-2.0 so that commercial medical software can use it. A
    CC BY-NC file in this repository would make that untrue while the headers
    still claimed otherwise, which is worse than not having the file.
    """
    for case in CASES:
        assert case["licence"].startswith("CC BY") and "NC" not in case["licence"], \
            f"{case['file']} is {case['licence']}"
        assert case["doi"], f"{case['file']} has no DOI to cite"
