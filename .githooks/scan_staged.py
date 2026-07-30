"""Refuse to commit anything that looks like a medical image.

Called by .githooks/pre-commit. Asks git for the staged files itself — being
handed them on stdin is what broke an earlier version of this check, which
passed everything silently because the heredoc carrying the script had already
consumed stdin. A guard that fails open is worse than no guard, because it is
still trusted; tests/test_hook.py exists to catch exactly that.

Identification is by content. A DICOM file named notes.dat is still a DICOM
file, and an extension filter would wave it through.
"""

import os
import subprocess
import sys

# Extensions where naming a transfer syntax UID is normal and expected.
SOURCE = {".py", ".md", ".c", ".h", ".toml", ".yml", ".yaml", ".txt",
          ".cfg", ".json", ".cff", ".sh", ".rst", ".ini"}

MEDICAL = {".dcm", ".ima", ".dicom", ".img", ".nii"}


def staged_files():
    out = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=AM"],
        capture_output=True, text=True, check=True).stdout
    return [line.strip() for line in out.splitlines() if line.strip()]


def why_blocked(path):
    """The reason this file may not be committed, or None."""
    extension = os.path.splitext(path)[1].lower()
    if extension in MEDICAL:
        return "medical image extension"

    try:
        with open(path, "rb") as handle:
            head = handle.read(4096)
    except OSError:
        return None

    # DICOM part-10: a 128-byte preamble, then the literal "DICM".
    if head[128:132] == b"DICM":
        return "DICOM part-10 file"

    # Raw DICOM with no preamble, and the bare transfer syntax UIDs. Source and
    # metadata may legitimately name a UID; a data file may not.
    if b"1.2.840.10008.1.2" in head and extension not in SOURCE:
        return "contains a DICOM transfer syntax UID"

    return None


def main():
    blocked = [(p, why) for p in staged_files()
               if os.path.isfile(p) and (why := why_blocked(p))]
    if not blocked:
        return 0

    print("\nCommit refused. These files look like medical images:\n",
          file=sys.stderr)
    for path, why in blocked:
        print(f"  {path}  ({why})", file=sys.stderr)
    print("\nPatient data must never enter this repository, and rewriting "
          "history\nafter a push does not remove it from a remote. If you are "
          "certain a\nfile is synthetic, commit it with --no-verify and say so "
          "in the message.\n", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
