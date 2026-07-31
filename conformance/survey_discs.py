r"""Count what a pile of DICOM discs actually contains, without reading a patient.

The real-file figures in README.md, CORRECTNESS.md and PAPER.md used to rest on
the author's word: the archive cannot be redistributed, and the script that
produced the numbers was not kept. An unreproducible number in a document whose
argument is reproducibility is the wrong kind of number, so this is that script,
and it now ships.

    python conformance/survey_discs.py /path/to/archive > survey.json

It reads device attributes and geometry — manufacturer, model, modality,
transfer syntax, bit depth, frame count — and nothing that describes a person.
Pixel data is never touched (`stop_before_pixels`), and files are identified by
the part-10 preamble rather than by extension, because an extension filter
misses every disc that names its files `IM_0001`.

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


def survey(root: str, progress=None) -> dict:
    import pydicom

    files = dicom = unreadable = 0
    frames = pixels = 0
    manufacturers: collections.Counter = collections.Counter()
    builds: collections.Counter = collections.Counter()
    modalities: collections.Counter = collections.Counter()
    syntaxes: collections.Counter = collections.Counter()
    depths: collections.Counter = collections.Counter()

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

    merged = None
    roots = argv[1:]
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
        result = survey(root, progress=sys.stderr)
        merged = result if merged is None else _merge(merged, result)

    print(json.dumps(merged, ensure_ascii=False, indent=1))
    return 0


def _merge(left: dict, right: dict) -> dict:
    out = dict(left)
    for key in ("files_walked", "dicom_files", "unreadable",
                "lossless_jpeg_frames", "pixels"):
        out[key] = left[key] + right[key]
    for key in ("manufacturers", "builds", "modalities", "transfer_syntaxes",
                "bits_stored"):
        counter = collections.Counter(dict(left[key]))
        counter.update(dict(right[key]))
        out[key] = counter.most_common()
    out["distinct_manufacturer_strings"] = len(out["manufacturers"])
    out["distinct_builds"] = len(out["builds"])
    out["distinct_models"] = len({key.split(" | ")[1] for key, _ in out["builds"]})
    return out


if __name__ == "__main__":
    sys.exit(main(sys.argv))
