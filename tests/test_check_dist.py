"""The release guard has to refuse, not just pass on the good case.

CI builds real artefacts and runs `tools/check_dist.py` over them, which proves
the checker accepts a correct distribution. It proves nothing about the case
the checker exists for, because CI never builds a broken one on purpose. This
does — a `py3-none-any` wheel, a Linux tag PyPI will not take, and an sdist
with its test suite missing are each constructed here and each must be refused.

The distinction matters because of how this project has failed before: a
regression test written for the C core passed against the unfixed code, and
nobody noticed until the fix was removed to check.
"""

import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
import check_dist                                          # noqa: E402


def _wheel(path: Path, name: str, members=("plumbline/__init__.py",)) -> Path:
    target = path / name
    with zipfile.ZipFile(target, "w") as archive:
        for member in members:
            archive.writestr(member, b"x")
    return target


def _sdist(path: Path, names) -> Path:
    target = path / "plumbline_dicom-0.1.0.tar.gz"
    with tarfile.open(target, "w:gz") as archive:
        for name in names:
            member = tarfile.TarInfo(f"plumbline_dicom-0.1.0/{name}")
            member.size = 0
            archive.addfile(member)
    return target


def test_it_refuses_a_pure_wheel(tmp_path):
    """The one `python -m build` produces, and the one that would ship."""
    problems = check_dist.check_wheel(
        _wheel(tmp_path, "plumbline_dicom-0.1.0-py3-none-any.whl"))
    assert problems
    assert "py3-none-any" in problems[0]


def test_it_refuses_a_platform_wheel_with_no_core(tmp_path):
    """Tagged for a platform, carrying nothing that needed the tag."""
    problems = check_dist.check_wheel(
        _wheel(tmp_path, "plumbline_dicom-0.1.0-py3-none-win_amd64.whl"))
    assert problems and "no _plumbline binary" in problems[0]


def test_it_refuses_a_tag_pypi_will_not_take(tmp_path):
    """`linux_x86_64` is what sysconfig reports and what Warehouse rejects."""
    problems = check_dist.check_wheel(
        _wheel(tmp_path, "plumbline_dicom-0.1.0-py3-none-linux_x86_64.whl",
               ("plumbline/__init__.py", "plumbline/_plumbline.so")))
    assert problems and "not one PyPI accepts" in problems[0]


@pytest.mark.parametrize("tag", ["manylinux_2_35_x86_64", "win_amd64",
                                 "macosx_11_0_arm64", "musllinux_1_2_x86_64"])
def test_it_accepts_a_wheel_that_keeps_the_promise(tmp_path, tag):
    suffix = ".so" if "linux" in tag else (".dll" if "win" in tag else ".dylib")
    assert not check_dist.check_wheel(
        _wheel(tmp_path, f"plumbline_dicom-0.1.0-py3-none-{tag}.whl",
               ("plumbline/__init__.py", f"plumbline/_plumbline{suffix}")))


def test_it_refuses_an_sdist_that_cannot_run_its_own_tests(tmp_path):
    """A packager who downloads the tarball to check it must be able to."""
    problems = check_dist.check_sdist(
        _sdist(tmp_path, ["src/plumbline/__init__.py", "native/plumbline.c",
                          "README.md", "LICENSE", "NOTICE"]))
    assert problems
    assert "tests/" in problems[0] and "conformance/" in problems[0]


def test_it_accepts_a_complete_sdist(tmp_path):
    assert not check_dist.check_sdist(
        _sdist(tmp_path, [f"{need}x" if need.endswith("/") else need
                          for need in check_dist.SDIST_NEEDS]))
