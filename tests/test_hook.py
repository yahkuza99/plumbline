"""The guard that keeps patient data out has to be tested like anything else.

It has already failed twice.

The first time, a rewrite that moved the scan into Python passed the file list
to the interpreter on stdin — which `python - <<EOF` had already spent on the
script itself — so the scan saw no files, found nothing, and let a decoy DICOM
file straight through. It exited 0 in under a second, which is what success
looks like from the outside.

The second time, the same check existed twice: a thorough version in the hook,
and a weaker one written separately in CI that looked only for a part-10
preamble. The weak copy was the one that always ran, because a hook is opt-in
and CI is not.

Everything below the first section is a bypass that worked against a version
of this scanner. They stay as tests because a closed hole reopens quietly.
"""

import gzip
import io
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

SCANNER = Path(__file__).parent.parent / ".githooks" / "scan_staged.py"

DICOM_PART10 = b"\x00" * 128 + b"DICM" + b"\x02\x00\x00\x00"
RAW_DICOM = b"1.2.840.10008.1.2.4.70" + b"\x00" * 200
FAR_UID = b"\x00" * 5000 + RAW_DICOM      # past the 4 KiB the scanner used to read
LFS_POINTER = (b"version https://git-lfs.github.com/spec/v1\n"
               b"oid sha256:ab\nsize 9\n")


def _zip_of(payload: bytes) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("scan.dcm", payload)
    return buffer.getvalue()


def _scan(tmp_path, files):
    """Stage `files` in a throwaway repository and run the real scanner."""
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    for name, data in files.items():
        (tmp_path / name).write_bytes(data)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    return subprocess.run([sys.executable, str(SCANNER)], cwd=tmp_path,
                          capture_output=True, text=True)


# --------------------------------------------------------------------------- #
# identified by content, not by name
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name, data, why", [
    ("scan.dcm", b"anything", "the extension alone"),
    ("scan_001.bin", DICOM_PART10, "DICM at offset 128, harmless extension"),
    ("notes.dat", RAW_DICOM, "a transfer syntax UID, harmless extension"),
    ("no_extension_at_all", DICOM_PART10, "how discs actually name files"),
])
def test_it_refuses(tmp_path, name, data, why):
    result = _scan(tmp_path, {name: data})
    assert result.returncode == 1, f"let through a file identifiable by {why}"
    assert name in result.stderr


def test_it_refuses_even_when_mixed_with_clean_files(tmp_path):
    """The failure that started this: a scan that reports nothing looks
    identical to a scan that found nothing."""
    result = _scan(tmp_path, {
        "README.md": b"ordinary\n",
        "frame.jpg": b"\xff\xd8" + b"\x00" * 64,
        "quietly.bin": DICOM_PART10,
    })
    assert result.returncode == 1
    assert "quietly.bin" in result.stderr


# --------------------------------------------------------------------------- #
# bypasses that worked
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name, data, bypass", [
    ("padded.bin", FAR_UID,
     "a UID further into the file than the scanner used to read"),
    ("notes.json", RAW_DICOM,
     "a source extension, which used to exempt the UID check outright"),
    ("helper.py", RAW_DICOM, "the same trick in a .py"),
    ("export.zip", _zip_of(DICOM_PART10), "an archive hiding the bytes"),
    ("export.gz", gzip.compress(DICOM_PART10), "a gzip stream"),
    ("big.dcmlfs", LFS_POINTER,
     "a git-lfs pointer, whose real bytes never enter history"),
    ("volume.raw", b"\x01\x02", "the companion file of an MHD volume"),
    ("brain.nii", b"\x01\x02", "NIfTI — not DICOM, still a patient"),
    ("disc.iso", b"\x01\x02", "a whole disc image"),
])
def test_it_refuses_known_bypasses(tmp_path, name, data, bypass):
    result = _scan(tmp_path, {name: data})
    assert result.returncode == 1, f"let through {bypass}"
    assert name in result.stderr


# --------------------------------------------------------------------------- #
# and still lets ordinary work through
# --------------------------------------------------------------------------- #

def test_source_naming_a_uid_still_commits(tmp_path):
    """This exemption has to keep working.

    A guard that cries wolf trains people into --no-verify, and that switches
    it off for everything else in the same commit — including whatever they
    had not noticed was still staged.
    """
    result = _scan(tmp_path, {
        "reference.py": b'LOSSLESS = ("1.2.840.10008.1.2.4.70",)\n',
        "manifest.json": b'{"syntax": "1.2.840.10008.1.2.4.57"}\n',
        "README.md": b"decodes 1.2.840.10008.1.2.4.70\n",
        "frame.jpg": b"\xff\xd8\xff\xc3" + b"\x00" * 64,
    })
    assert result.returncode == 0, result.stderr


def test_it_refuses_rather_than_failing_open(tmp_path):
    """Run outside a repository, the scanner must not report success.

    Every failure this guard has had looked like success from the outside.
    """
    result = subprocess.run([sys.executable, str(SCANNER)], cwd=tmp_path,
                            capture_output=True, text=True)
    assert result.returncode != 0


# --------------------------------------------------------------------------- #
# The history scan is the mandatory gate — CI runs it and a hook is opt-in —
# and until these were written it had no test at all. Both cases below got a
# file past it.
# --------------------------------------------------------------------------- #

def _repo(tmp_path):
    for command in (["git", "init", "-q"],
                    ["git", "config", "user.email", "t@example.invalid"],
                    ["git", "config", "user.name", "t"]):
        subprocess.run(command, cwd=tmp_path, check=True)


def _history_scan(tmp_path):
    return subprocess.run([sys.executable, str(SCANNER), "--history"],
                          cwd=tmp_path, capture_output=True, text=True)


def test_history_finds_a_blob_with_no_path(tmp_path):
    """A tag can point straight at a blob, which then has no filename.

    The scan used to index blobs by path and never read this one at all.
    """
    _repo(tmp_path)
    (tmp_path / "seed.txt").write_bytes(b"seed\n")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "seed"], cwd=tmp_path, check=True)

    (tmp_path / "loose").write_bytes(DICOM_PART10)
    sha = subprocess.run(["git", "hash-object", "-w", "loose"], cwd=tmp_path,
                         capture_output=True, text=True, check=True).stdout.strip()
    (tmp_path / "loose").unlink()
    subprocess.run(["git", "tag", "-a", "t", "-m", "m", sha], cwd=tmp_path, check=True)

    assert _history_scan(tmp_path).returncode == 1


def test_history_judges_a_blob_under_every_name_it_ever_had(tmp_path):
    """Renaming afterwards must not launder a blob.

    Committed as .dcm, renamed to .txt, the scan used to see only the latest
    name and let it through for ever.
    """
    _repo(tmp_path)
    (tmp_path / "scan.dcm").write_bytes(RAW_DICOM)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "one", "--no-verify"],
                   cwd=tmp_path, check=True)
    subprocess.run(["git", "mv", "scan.dcm", "notes.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "two", "--no-verify"],
                   cwd=tmp_path, check=True)

    result = _history_scan(tmp_path)
    assert result.returncode == 1
    assert "scan.dcm" in result.stderr


def test_a_rename_is_still_scanned(tmp_path):
    """Git calls a sufficiently similar add-plus-delete a rename, and the
    staged-file filter used to skip those."""
    _repo(tmp_path)
    (tmp_path / "config.cfg").write_bytes(b"# setting\n" * 200)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=tmp_path, check=True)

    (tmp_path / "scan.dcm").write_bytes(b"# setting\n" * 200 + RAW_DICOM)
    (tmp_path / "config.cfg").unlink()
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)

    result = subprocess.run([sys.executable, str(SCANNER)], cwd=tmp_path,
                            capture_output=True, text=True)
    assert result.returncode == 1, "a rename slipped past the staged-file filter"
    assert "scan.dcm" in result.stderr
