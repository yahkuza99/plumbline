"""Refuse to let anything that looks like a medical image into this repository.

One implementation, three callers:

    python .githooks/scan_staged.py             # staged files (the pre-commit hook)
    python .githooks/scan_staged.py --history   # every blob ever committed (CI)
    python .githooks/scan_staged.py --paths a b # named files

That matters more than it sounds. This check previously existed twice — a
thorough version in the hook, and a weaker one written separately in the CI
workflow that looked only for a part-10 preamble. The weak copy was the one
that always ran, because a hook is opt-in and CI is not, so the mandatory gate
was the one that missed raw DICOM. Two copies of a rule drift, and the copy
that drifts is found by the thing it was supposed to prevent.

Identification is by content wherever possible. A DICOM file named `notes.dat`
is still a DICOM file, and an extension filter waves it through.

Every rule below that is not obvious was added because red-teaming this
scanner got a synthetic DICOM file past the previous version of it.

A guard that fails open is worse than no guard, because the project keeps
relying on it. This one exits non-zero on any error it did not expect, and
tests/test_hook.py stages decoys against it on every commit.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

# Extensions where naming a transfer syntax UID is expected: source, metadata,
# documentation. The exemption is checked against the content as well — see
# `_looks_like_text`.
SOURCE = {".py", ".md", ".c", ".h", ".toml", ".yml", ".yaml", ".txt",
          ".cfg", ".json", ".cff", ".sh", ".rst", ".ini"}

MEDICAL = {".dcm", ".ima", ".dicom", ".img", ".nii", ".mha", ".mhd",
           ".raw", ".hdr", ".zraw", ".iso", ".nrrd", ".analyze"}

PREAMBLE = b"DICM"              # part-10, at offset 128
UID_STEM = b"1.2.840.10008"     # the DICOM UID root — all of them, not just pixel syntaxes
LFS_POINTER = b"version https://git-lfs.github.com/spec/v1"

# Read this much rather than a fixed small window. An export can carry a large
# private tag before it reaches the metadata that names it, and a 4 KiB window
# let a whole DICOM file through behind nothing but padding.
SCAN_BYTES = 8 << 20

# An archive hides its contents from a byte scan. Decompressing input from an
# untrusted source to look inside invites a decompression bomb, and nothing in
# this project needs an archive committed, so they are refused outright.
ARCHIVE_MAGIC = [
    (b"PK\x03\x04", "zip archive"),
    (b"PK\x05\x06", "empty zip archive"),
    (b"\x1f\x8b", "gzip stream"),
    (b"BZh", "bzip2 stream"),
    (b"\xfd7zXZ\x00", "xz stream"),
    (b"7z\xbc\xaf\x27\x1c", "7-zip archive"),
    (b"Rar!\x1a\x07", "rar archive"),
    (b"\x28\xb5\x2f\xfd", "zstd stream"),
]


def _looks_like_text(data: bytes) -> bool:
    """Source and metadata are text. A renamed DICOM file is not.

    The extension whitelist exists so that source naming a transfer syntax UID
    can be committed. Keying the exemption on the extension alone meant a raw
    DICOM file called `notes.json` was exempt too. Requiring the bytes to be
    text as well closes that without making the whitelist useless.
    """
    sample = data[:65536]
    if b"\x00" in sample:
        return False
    try:
        sample.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def verdict(name: str, head: bytes) -> str | None:
    """Why this content may not be committed under this name, or None.

    `name` contributes only its extension, and only ever to make the answer
    stricter — never to excuse content that would otherwise be refused, apart
    from the text-file case, which is itself checked against the bytes.
    """
    lowered = name.lower()
    for extension in MEDICAL:
        if lowered.endswith(extension):
            return "medical image extension"

    if head[128:132] == PREAMBLE:
        return "DICOM part-10 file"

    if head.startswith(LFS_POINTER):
        return ("git-lfs pointer — the real bytes are not in this history, "
                "so they cannot be checked")

    for magic, kind in ARCHIVE_MAGIC:
        if head.startswith(magic):
            return f"{kind} — its contents cannot be checked, so it is refused"
    if head[257:262] == b"ustar":
        return "tar archive — its contents cannot be checked, so it is refused"

    if UID_STEM in head:
        # Text may name a UID; that is what source and manifests do. Binary
        # carrying one came out of a scanner.
        if not (os.path.splitext(lowered)[1] in SOURCE and _looks_like_text(head)):
            return "contains a DICOM UID"

    return None


def staged() -> list[str]:
    out = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=AM"],
        capture_output=True, text=True, check=True).stdout
    return [line.strip() for line in out.splitlines() if line.strip()]


def scan_worktree(paths) -> list[tuple[str, str]]:
    blocked = []
    for path in paths:
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "rb") as handle:
                head = handle.read(SCAN_BYTES)
        except OSError as error:
            blocked.append((path, f"unreadable, so unverifiable: {error}"))
            continue
        why = verdict(path, head)
        if why:
            blocked.append((path, why))
    return blocked


def scan_history() -> list[tuple[str, str]]:
    """Every blob reachable from any ref, under the path it was committed as.

    Rewriting history does not remove a blob from a remote that has already
    seen it, so this is the check that has to be right.
    """
    listing = subprocess.run(["git", "rev-list", "--objects", "--all"],
                             capture_output=True, text=True, check=True).stdout
    names = {}
    for line in listing.splitlines():
        sha, _, path = line.partition(" ")
        if path:
            names[sha] = path
    if not names:
        return []

    kinds = subprocess.run(
        ["git", "cat-file", "--batch-check=%(objecttype) %(objectname)"],
        input="\n".join(names), capture_output=True, text=True, check=True).stdout
    blobs = [line.split()[1] for line in kinds.splitlines()
             if line.startswith("blob ")]
    if not blobs:
        return []

    blocked = []
    batch = subprocess.Popen(["git", "cat-file", "--batch"],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        for sha in blobs:
            batch.stdin.write((sha + "\n").encode())
            batch.stdin.flush()
            header = batch.stdout.readline().split()
            if len(header) < 3:
                continue
            size = int(header[2])
            body = batch.stdout.read(size)
            batch.stdout.read(1)                        # trailing newline
            why = verdict(names.get(sha, sha), body[:SCAN_BYTES])
            if why:
                blocked.append((f"{names.get(sha, sha)}  ({sha[:10]})", why))
    finally:
        batch.stdin.close()
        batch.wait()
    return blocked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", action="store_true",
                        help="scan every blob ever committed, not the index")
    parser.add_argument("--paths", nargs="*", help="scan these files")
    args = parser.parse_args()

    if args.history:
        blocked, scope = scan_history(), "this repository's history"
    else:
        blocked = scan_worktree(args.paths if args.paths else staged())
        scope = "the files being committed"

    if not blocked:
        return 0

    print(f"\nRefused. These files in {scope} look like medical images:\n",
          file=sys.stderr)
    for path, why in blocked:
        print(f"  {path}  ({why})", file=sys.stderr)
    print("\nPatient data must never enter this repository, and rewriting\n"
          "history after a push does not remove it from a remote.\n"
          "\n"
          "If you are certain a file is synthetic, commit it with --no-verify\n"
          "and say so in the message — then check `git status` first. A file\n"
          "refused here stays staged, and the next --no-verify will take it.\n",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:                  # never fail open
        print(f"\nThe medical-image scan could not run: {error!r}\n"
              "Refusing rather than allowing an unchecked commit.\n",
              file=sys.stderr)
        sys.exit(2)
