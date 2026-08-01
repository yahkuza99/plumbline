r"""Count what a pile of DICOM discs actually contains, without reading a patient.

The real-file figures in README.md, CORRECTNESS.md and PAPER.md used to rest on
the author's word: the archive cannot be redistributed, and the script that
produced the numbers was not kept. An unreproducible number in a document whose
argument is reproducibility is the wrong kind of number, so this is that script,
and it now ships.

    python conformance/survey_discs.py /path/to/archive > survey.json

It reads device attributes and geometry — manufacturer, model, modality,
transfer syntax, bit depth, frame count — and nothing that describes a person.
Files are identified by the part-10 preamble rather than by extension, because
an extension filter misses every disc that names its files `IM_0001`.

By default no pixel data is read at all (`stop_before_pixels`). With
`--predictors` the first 256 KiB of each file is read so the JPEG scan header
can be parsed, because the predictor is recorded there and in no DICOM tag —
and "every real frame uses predictor 1" is the sharpest limitation this project
states. Even then the encoded samples themselves are never decoded, and the
output is unchanged in kind: counts, never content.

**No path or filename is printed, ever, including on failure.** Archives like
this are routinely organised into folders named after the patient, so a
traceback quoting a path is a disclosure. Failures are counted; they are never
located. Read that as the house style rather than as paranoia — the same rule
is why `.githooks/scan_staged.py` exists.

On Windows the walk is prefixed with `\\?\`, without which `os.scandir` — and
`dir /s`, which is how this was first attempted — silently stops at the paths
longer than 260 characters that a deep disc archive is full of, and undercounts
by an amount nobody can see.

Output is one JSON object on stdout, and nothing is written anywhere.
"""

from __future__ import annotations

import collections
import json
import os
import sys

# The two lossless-JPEG transfer syntaxes this project decodes.
LOSSLESS = {
    "1.2.840.10008.1.2.4.57": "process 14",
    "1.2.840.10008.1.2.4.70": "process 14, selection value 1",
}


def _long(path: str) -> str:
    """Step past MAX_PATH on Windows; a no-op everywhere else."""
    absolute = os.path.abspath(path)
    if os.name == "nt" and not absolute.startswith("\\\\?\\"):
        return "\\\\?\\" + absolute
    return absolute


def walk(top: str):
    """Every file below `top`, surviving directories that cannot be read.

    Iterative rather than recursive: these archives nest deeply enough to
    exhaust the interpreter's stack, and `os.walk` gives no way to carry on
    past a directory the account cannot open.
    """
    stack = [top]
    while stack:
        here = stack.pop()
        try:
            with os.scandir(here) as entries:
                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(entry.path)
                        else:
                            yield entry.path
                    except OSError:
                        pass                    # unreadable entry, counted below
        except OSError:
            pass


# How much of a file to read when the predictor is wanted. The JPEG stream
# begins right after the DICOM header, and SOF3 and SOS sit within a few
# hundred bytes of its SOI, so this reaches them without reading the pixels.
HEADER_WINDOW = 256 << 10


def frame_parameters(head: bytes) -> dict | None:
    """The SOF3 and SOS fields of the first encapsulated frame, or None.

    A small JPEG marker walk rather than a call into plumbline: this script has
    to run on a machine holding the discs, which is not the machine the library
    is installed on, and a survey tool that needs the thing it is surveying for
    installed alongside it is a survey tool nobody runs.

    Returns the predictor — the Ss field of SOS, which for lossless JPEG is the
    predictor selector 1-7 — with the precision, component count and point
    transform that go with it. None where the markers are not in `head`.
    """
    start = head.find(b"\xff\xd8\xff")           # SOI of the first fragment
    if start < 0:
        return None

    at = start + 2
    found = {}
    while at + 3 < len(head):
        if head[at] != 0xFF:
            at += 1
            continue
        while at + 1 < len(head) and head[at + 1] == 0xFF:
            at += 1                              # fill bytes, T.81 B.1.1.3
        marker = head[at + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            at += 2
            continue
        if at + 3 >= len(head):
            break
        length = (head[at + 2] << 8) | head[at + 3]
        body = head[at + 4:at + 2 + length]
        if marker == 0xC3 and len(body) >= 6:               # SOF3
            found["precision"] = body[0]
            found["components"] = body[5]
        elif marker == 0xDD and len(body) >= 2:             # DRI
            found["restart_interval"] = (body[0] << 8) | body[1]
        elif marker == 0xDA:                                # SOS
            count = body[0] if body else 0
            tail = 1 + count * 2
            if len(body) >= tail + 3:
                found["predictor"] = body[tail]
                found["point_transform"] = body[tail + 2] & 0x0F
            return found if "predictor" in found else None
        at += 2 + length
    return None


def survey(root: str, progress=None, predictors: bool = False) -> dict:
    import pydicom

    files = dicom = unreadable = 0
    frames = pixels = 0
    manufacturers: collections.Counter = collections.Counter()
    builds: collections.Counter = collections.Counter()
    modalities: collections.Counter = collections.Counter()
    syntaxes: collections.Counter = collections.Counter()
    depths: collections.Counter = collections.Counter()
    predictor_counts: collections.Counter = collections.Counter()
    combinations: collections.Counter = collections.Counter()
    unparsed = 0

    for path in walk(_long(root)):
        files += 1
        if progress and files % 20_000 == 0:
            print(f"  ...{files:,} files, {dicom:,} DICOM", file=progress, flush=True)

        try:
            with open(path, "rb") as handle:
                if handle.read(132)[128:] != b"DICM":
                    continue
        except OSError:
            unreadable += 1
            continue

        dicom += 1
        try:
            data = pydicom.dcmread(path, stop_before_pixels=True)
        except Exception:                       # never quote the path
            unreadable += 1
            continue

        syntax = str(getattr(data.file_meta, "TransferSyntaxUID", "") or "")
        syntaxes[syntax] += 1
        if syntax not in LOSSLESS:
            continue

        maker = str(data.get("Manufacturer", "") or "(untagged)").strip()
        model = str(data.get("ManufacturerModelName", "") or "(untagged)").strip()
        modality = str(data.get("Modality", "") or "(untagged)").strip()
        count = int(data.get("NumberOfFrames", 1) or 1)

        manufacturers[maker] += count
        builds[f"{maker} | {model} | {modality}"] += count
        modalities[modality] += count
        depths[int(data.get("BitsStored", 0) or 0)] += count
        frames += count
        pixels += int(data.get("Rows", 0) or 0) * int(data.get("Columns", 0) or 0) * count

        if predictors:
            # The predictor lives in the scan header, not in any DICOM tag, so
            # this is the one figure that costs a look at the encoded stream.
            # It is counted because "every real frame uses predictor 1" is the
            # sharpest limitation this project states and nothing measured it.
            #
            # Per file, not per frame: a multi-frame instance shares one scan
            # header across its frames, and claiming otherwise would inflate
            # the count by exactly the multi-frame instances.
            # Read the window now rather than up front. Only the lossless
            # files need it, and reading 256 KiB of every file in the archive
            # to reach the two in three that are lossless turned a thirteen
            # minute survey into a multi-hour one.
            try:
                with open(path, "rb") as handle:
                    head = handle.read(HEADER_WINDOW)
            except OSError:
                head = b""

            found = frame_parameters(head)
            if found is None:
                unparsed += 1
            else:
                predictor_counts[found.get("predictor", 0)] += 1
                # Restart intervals are recorded as MCU-rows rather than raw
                # MCU, because the claim being tested is "every frame that
                # restarts, restarts once per row" and a raw interval of 852
                # says nothing without the width beside it. T.81 §H.1.1 makes
                # the interval a whole number of rows, so this divides exactly
                # — where it does not, the frame is one this decoder refuses,
                # and `rows?` records that rather than rounding it away.
                interval = found.get("restart_interval", 0)
                width = int(data.get("Columns", 0) or 0)
                if not interval:
                    rows = "none"
                elif width and interval % width == 0:
                    rows = str(interval // width)
                else:
                    rows = "not-whole-rows"
                combinations[
                    f"P{found.get('precision')} pred{found.get('predictor')} "
                    f"comp{found.get('components')} "
                    f"pt{found.get('point_transform')} "
                    f"restart-rows:{rows}"] += 1

    return {
        "files_walked": files,
        "dicom_files": dicom,
        "unreadable": unreadable,
        "lossless_jpeg_frames": frames,
        "pixels": pixels,
        "distinct_manufacturer_strings": len(manufacturers),
        "distinct_builds": len(builds),
        "distinct_models": len({key.split(" | ")[1] for key in builds}),
        "manufacturers": manufacturers.most_common(),
        "builds": builds.most_common(),
        "modalities": modalities.most_common(),
        "bits_stored": sorted(depths.items()),
        "transfer_syntaxes": syntaxes.most_common(),
        # Present only with --predictors; counted per file rather than per
        # frame, and `predictors_unparsed` is the number whose scan header was
        # not inside the window read. It is reported rather than dropped: a
        # survey that could not look at something must not read as one that
        # looked and found nothing.
        "predictors": sorted(predictor_counts.items()),
        "predictors_unparsed": unparsed,
        "combinations": combinations.most_common(),
        # No predictor tally: SOF3 carries it, and reading SOF3 means
        # reading pixel data. That the real frames are all predictor 1
        # is a claim this script cannot support, and it does not
        # pretend to by shipping a field that is always empty.
    }


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__.strip().splitlines()[0], file=sys.stderr)
        print(f"usage: {argv[0]} <archive-root> [more roots...]", file=sys.stderr)
        return 2

    predictors = "--predictors" in argv
    roots = [a for a in argv[1:] if not a.startswith("--")]
    if not roots:
        print(f"usage: {argv[0]} [--predictors] <archive-root> [more roots...]",
              file=sys.stderr)
        return 2

    merged = None
    for number, root in enumerate(roots, start=1):
        if not os.path.isdir(root):
            # By position, not by name. The rule at the top of this file is
            # that no path is printed, and a root someone points at a patient's
            # folder is still a path — even though they typed it themselves,
            # because what gets typed into a terminal ends up in shell history,
            # in a CI log, and in the issue where they paste the output.
            print(f"argument {number} is not a directory", file=sys.stderr)
            return 2
        print(f"surveying root {number} of {len(roots)}", file=sys.stderr)
        result = survey(root, progress=sys.stderr, predictors=predictors)
        merged = result if merged is None else _merge(merged, result)

    print(json.dumps(merged, ensure_ascii=False, indent=1))
    return 0


def _merge(left: dict, right: dict) -> dict:
    out = dict(left)
    for key in ("files_walked", "dicom_files", "unreadable",
                "lossless_jpeg_frames", "pixels", "predictors_unparsed"):
        out[key] = left[key] + right[key]
    for key in ("manufacturers", "builds", "modalities", "transfer_syntaxes",
                "bits_stored", "predictors", "combinations"):
        counter = collections.Counter(dict(left[key]))
        counter.update(dict(right[key]))
        out[key] = counter.most_common()
    out["distinct_manufacturer_strings"] = len(out["manufacturers"])
    out["distinct_builds"] = len(out["builds"])
    out["distinct_models"] = len({key.split(" | ")[1] for key, _ in out["builds"]})
    return out


if __name__ == "__main__":
    sys.exit(main(sys.argv))
