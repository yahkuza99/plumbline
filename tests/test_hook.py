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
    tmp_path.mkdir(parents=True, exist_ok=True)   # callers pass subdirectories
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    for name, data in files.items():
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    return subprocess.run([sys.executable, str(SCANNER)], cwd=tmp_path,
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace")


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
        "conformance/corpus/pt1_p8.jpg": b"\xff\xd8\xff\xc3" + b"\x00" * 64,
    })
    assert result.returncode == 0, result.stderr


def test_a_lossless_frame_outside_the_corpus_is_refused(tmp_path):
    """The same bytes, one directory over, must not commit.

    This test previously asserted the opposite — that `frame.jpg` at the root
    commits cleanly — and it was the assertion, not an oversight, that kept the
    hole open. Pulling the encapsulated frame out of a study that decodes wrong
    and saving it next to the code is the obvious way to debug this project.
    That frame is the patient's chest with the header stripped off, and no rule
    here could ever recognise it as one, because it is byte-for-byte the kind of
    file the repository legitimately holds 1,713 of.

    So the rule is location, not content: images are test material where the
    project keeps its test material, and refused everywhere else. It is a
    coarse rule that costs nothing today, and it is the only one that can catch
    this at all.
    """
    frame = b"\xff\xd8\xff\xc3" + b"\x00" * 64
    assert _scan(tmp_path / "loose", {"frame.jpg": frame}).returncode == 1
    assert _scan(tmp_path / "png", {"docs/example.png":
                                    b"\x89PNG\r\n\x1a\n" + b"\x00" * 64}).returncode == 1
    assert _scan(tmp_path / "corpus", {"tests/data/idc/philips_mx8000_ct.jpg":
                                       frame}).returncode == 0


# --------------------------------------------------------------------------- #
# Round two. The bypasses above were found by attacking the scanner; these were
# found by attacking it again after it had been fixed, and every one of them is
# something a person does by accident rather than something an attacker does on
# purpose. That is what makes them the dangerous kind.
# --------------------------------------------------------------------------- #

def test_a_thai_path_is_scanned_like_any_other(tmp_path):
    """The bypass that mattered most, because this project is written in Thai.

    Git quotes any path outside ASCII, so `ผู้ป่วย/scan.dcm` arrived as the
    literal 25-character string `"\\340\\270\\234..."`. That names no file on
    disk, `os.path.isfile` said so, and the loop moved on without a word. The
    identical file under an ASCII name was refused, which is how the hole
    stayed invisible: every test anyone wrote happened to use ASCII.

    A Thai folder name is not an edge case here. It is the author's own
    keyboard.
    """
    for index, folder in enumerate(("ผู้ป่วย", "café", "研究データ", "📁")):
        result = _scan(tmp_path / f"repo{index}",
                       {f"{folder}/scan.dcm": DICOM_PART10 + RAW_DICOM})
        assert result.returncode == 1, f"{folder}/ was skipped: {result.stderr}"


def test_it_judges_the_index_not_the_worktree(tmp_path):
    """Staging a file and then deleting it does not unstage it.

    `git add .` picks up an export, you notice, you delete the file, you
    commit. The blob is still in the index and still goes into the commit. The
    scanner read the worktree, found nothing where the path pointed, and
    reported clean — so the one moment the mistake was still recoverable passed
    in silence.
    """
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "scan.dcm").write_bytes(DICOM_PART10 + RAW_DICOM)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    (tmp_path / "scan.dcm").unlink()                  # deleted, but still staged

    result = subprocess.run([sys.executable, str(SCANNER)], cwd=tmp_path,
                            capture_output=True, text=True)
    assert result.returncode == 1, (
        "the blob is in the index and will be committed; deleting the file "
        f"from the worktree did not remove it. stderr={result.stderr!r}")


def test_it_judges_the_bytes_staged_not_the_bytes_now_on_disk(tmp_path):
    """Editing a file after staging it leaves the staged version untouched."""
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "notes.txt").write_bytes(DICOM_PART10 + RAW_DICOM)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    (tmp_path / "notes.txt").write_bytes(b"just some notes\n")   # innocent now

    result = subprocess.run([sys.executable, str(SCANNER)], cwd=tmp_path,
                            capture_output=True, text=True)
    assert result.returncode == 1, result.stderr


def test_the_dicom_json_and_xml_models_are_refused(tmp_path):
    """A whole study written out as text passed every content rule.

    `dcm2json` and `dcm2xml` produce exactly this, and a decoder project's
    fixtures are exactly this shape. Name, hospital number, date of birth and
    the pixels in InlineBinary — all valid UTF-8, no NUL byte, no part-10
    preamble — so `_looks_like_text` declared it source and exempted it.

    The identifiers below are invented.
    """
    study = (b'{"00100010":{"vr":"PN","Value":[{"Alphabetic":"SYNTHETIC^CASE"}]},'
             b'"00100020":{"vr":"LO","Value":["HN0000001"]},'
             b'"00100030":{"vr":"DA","Value":["19551103"]},'
             b'"7FE00010":{"vr":"OB","InlineBinary":"AAAA"}}')
    assert _scan(tmp_path / "json", {"study.json": study}).returncode == 1

    xml = (b'<?xml version="1.0"?><NativeDicomModel>'
           b'<DicomAttribute tag="00100010" vr="PN" keyword="PatientName">'
           b'<PersonName number="1"><Alphabetic><FamilyName>SYNTHETIC'
           b'</FamilyName></Alphabetic></PersonName></DicomAttribute>'
           b'</NativeDicomModel>')
    assert _scan(tmp_path / "xml", {"study.xml": xml}).returncode == 1

    # And source that quotes one still commits. The first version of this rule
    # refused this very file, because a test that carries a study as a literal
    # matches the same pattern the study does — so the rule checks that the
    # content *is* such a document, not merely that it mentions one.
    quoting = b'STUDY = ' + study + b'\nassert scan(STUDY) == "refused"\n'
    assert _scan(tmp_path / "src", {"test_x.py": quoting}).returncode == 0


def test_text_in_front_of_a_dicom_does_not_exempt_it(tmp_path):
    """The window that decides the exemption must cover what it exempts.

    `_looks_like_text` sampled 64 KiB while the UID search covered 8 MiB, so
    anything with 64 KiB of clean ASCII in front of it — a de-identification
    audit log, an export manifest, an XML sidecar — was declared text, and
    every binary byte behind it went uninspected. An anonymiser that writes its
    own receipt ahead of the instance produces this file by default.
    """
    padded = b"A" * 70000 + DICOM_PART10 + RAW_DICOM
    assert _scan(tmp_path, {"export.bin": padded}).returncode == 1


def test_a_submodule_is_reported_rather_than_skipped(tmp_path):
    """No patient bytes enter this object store — and that is not the point.

    `git submodule add /d/xray vendor` is a convenient way to wire a local
    imaging corpus into a checkout, and it committed with no message at all.
    A recursive clone then fetches it. The scanner cannot see inside a gitlink,
    so it has to say that, because a scan that cannot check something must
    never look identical to one that checked and found nothing.
    """
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=corpus, check=True)
    (corpus / "scan.dcm").write_bytes(DICOM_PART10)
    subprocess.run(["git", "add", "-A"], cwd=corpus, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t",
                    "commit", "-qm", "corpus"], cwd=corpus, check=True)

    work = tmp_path / "work"
    work.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=work, check=True)
    added = subprocess.run(
        ["git", "-c", "protocol.file.allow=always", "submodule", "add", "-q",
         str(corpus), "vendor"], cwd=work, capture_output=True, text=True)
    if added.returncode != 0:
        pytest.skip(f"git refused the local submodule: {added.stderr.strip()}")

    result = subprocess.run([sys.executable, str(SCANNER)], cwd=work,
                            capture_output=True, text=True)
    assert result.returncode == 1
    assert "submodule" in result.stderr


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
