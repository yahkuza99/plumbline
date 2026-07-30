"""The guard that keeps patient data out has to be tested like anything else.

It has already failed once. A rewrite that moved the scan into Python passed
the file list to the interpreter on stdin — which `python - <<EOF` had already
spent on the script itself — so the scan saw no files, found nothing, and let a
decoy DICOM file straight through. It looked like it worked. It exited 0
quickly, which is what success looks like from the outside.

A guard that fails open is worse than no guard, because the project keeps
relying on it. So the decoys live here and run on every commit.
"""

import subprocess
import sys
from pathlib import Path

import pytest

SCANNER = Path(__file__).parent.parent / ".githooks" / "scan_staged.py"


def _scan(tmp_path, files):
    """Stage `files` in a throwaway repository and run the real scanner."""
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    for name, data in files.items():
        (tmp_path / name).write_bytes(data)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    return subprocess.run([sys.executable, str(SCANNER)], cwd=tmp_path,
                          capture_output=True, text=True)


DICOM_PART10 = b"\x00" * 128 + b"DICM" + b"\x02\x00\x00\x00"
RAW_DICOM = b"1.2.840.10008.1.2.4.70" + b"\x00" * 200


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


def test_it_allows_ordinary_files(tmp_path):
    """A UID in source or metadata is normal; blocking it would train people
    to pass --no-verify, which disables the guard for everything else too."""
    result = _scan(tmp_path, {
        "reference.py": b'LOSSLESS = ("1.2.840.10008.1.2.4.70",)\n',
        "manifest.json": b'{"syntax": "1.2.840.10008.1.2.4.57"}\n',
        "README.md": b"decodes 1.2.840.10008.1.2.4.70\n",
        "frame.jpg": b"\xff\xd8\xff\xc3" + b"\x00" * 64,
    })
    assert result.returncode == 0, result.stderr


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
