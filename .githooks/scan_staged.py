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

    if UID_STEM in head and not _looks_like_text(head):
        # Source, documentation and manifests name UIDs; that is their job,
        # and this project's do it constantly. What comes out of a scanner is
        # binary. Deciding on the bytes rather than the extension closes the
        # hole where a raw export called `notes.json` was exempt, and stops
        # the hook script — which quotes a UID and has no extension at all —
        # being reported as a patient file.
        return "contains a DICOM UID"

    return None


def staged() -> list[str]:
    """Every path whose content this commit introduces.

    The filter is not `AM`. Git reports a file as `R` when it is similar
    enough to one that disappeared, and `--diff-filter=AM` skips those — so
    deleting `notes.cfg` and adding `scan.dcm` with the same bulk of bytes
    produced a rename, and a file named `.dcm` walked past this scanner
    untouched. Confirmed against git's own rename detection, which is on by
    default and needs no argument to trigger.
    """
    out = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMRT"],
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


def _every_blob() -> list[str]:
    """Every blob a fetch could reach, plus everything the reflog can restore.

    Two things this deliberately is and is not:

    * It includes `--reflog`, so an old stash entry or a commit dropped by
      `reset --hard` or `--amend` is still examined. Those are recoverable
      and were invisible to a plain `--all`.
    * It reads shas rather than indexing by path. A tag can point straight at
      a blob, which then has no filename — and the previous version keyed the
      whole scan on the path, so that blob was never read at all.

    It stops short of `--batch-all-objects`, which also reports loose objects
    no ref or reflog can reach. Those never leave the machine — `git push`
    sends reachable objects only — and reporting them means reporting every
    experiment anyone has ever abandoned.
    """
    out = subprocess.run(
        ["git", "rev-list", "--objects", "--all", "--reflog", "--remotes"],
        capture_output=True, text=True, check=True).stdout
    shas = [line.split()[0] for line in out.splitlines() if line.strip()]
    if not shas:
        return []
    kinds = subprocess.run(
        ["git", "cat-file", "--batch-check=%(objecttype) %(objectname)"],
        input="\n".join(shas), capture_output=True, text=True, check=True).stdout
    return [line.split()[1] for line in kinds.splitlines() if line.startswith("blob ")]


def _blob_names() -> dict[str, set[str]]:
    """Every path each blob has ever been committed under.

    A blob keeps one identity and can have many names. Recording only the
    latest let a file committed as `scan.dcm` and renamed to `notes.txt`
    afterwards be judged under the harmless name for ever.
    """
    listing = subprocess.run(
        ["git", "cat-file", "--batch-all-objects",
         "--batch-check=%(objecttype) %(objectname)"],
        capture_output=True, text=True, check=True).stdout
    trees = [line.split()[1] for line in listing.splitlines()
             if line.startswith("tree ")]

    # Every tree, rather than `git rev-list --objects --all`, which prints a
    # blob once and attaches whichever path it happened to meet first. A file
    # committed as `scan.dcm` and renamed afterwards came back as `notes.txt`
    # alone, so a rule that keys on the extension never saw the name that
    # would have triggered it.
    names: dict[str, set[str]] = {}
    for tree in trees:
        entries = subprocess.run(["git", "ls-tree", tree],
                                 capture_output=True, text=True).stdout
        for line in entries.splitlines():
            head, _, name = line.partition("\t")
            parts = head.split()
            if len(parts) >= 3 and parts[1] == "blob" and name:
                names.setdefault(parts[2], set()).add(name)
    return names


def scan_history() -> list[tuple[str, str]]:
    """Every blob in the repository, under every name it has ever carried.

    Rewriting history does not remove a blob from a remote that has already
    seen it, so this is the check that has to be right.
    """
    blobs = _every_blob()
    if not blobs:
        return []
    names = _blob_names()

    blocked = []
    batch = subprocess.Popen(["git", "cat-file", "--batch"],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        for sha in blobs:
            batch.stdin.write((sha + "\n").encode())
            batch.stdin.flush()
            header = batch.stdout.readline().split()
            if len(header) < 3:
                # Unreadable object. Refuse rather than skip: a scan that
                # cannot see something must not report that it saw nothing.
                blocked.append((sha, "could not be read, so could not be checked"))
                continue
            body = batch.stdout.read(int(header[2]))
            batch.stdout.read(1)                        # trailing newline
            head = body[:SCAN_BYTES]
            aliases = sorted(names.get(sha, {sha}))
            reasons = [why for why in (verdict(name, head) for name in aliases) if why]
            if reasons:
                # Report every name the blob has carried, not the first that
                # matched. Whoever has to find and purge it needs the name it
                # was committed under, which is rarely the one it has now.
                where = aliases[0]
                if len(aliases) > 1:
                    where = f"{where}  (also: {', '.join(aliases[1:])})"
                blocked.append((f"{where}  ({sha[:10]})", reasons[0]))
    finally:
        batch.stdin.close()
        if batch.wait() != 0:
            blocked.append(("git cat-file", "exited non-zero; the scan is incomplete"))
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
