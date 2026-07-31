"""Refuse to publish a distribution that does not do what the README promises.

Run over `dist/` before uploading anything:

    python tools/check_dist.py dist/

The README's central practical claim is that a wheel carries the compiled core,
so nobody needs a C toolchain. Three ways that claim quietly becomes false, all
of which produce a build that exits 0:

* **A pure `py3-none-any` wheel.** `python -m build` with no arguments builds
  the sdist first and then builds the wheel *from that sdist*, which cannot
  contain a binary. The result is a wheel with no core — and because `any`
  matches every platform, pip would hand it to every user in preference to
  nothing, at a few hundredth of the promised speed. The build says so in one
  warning line among a screenful of output.

* **A wheel tagged for a platform PyPI will not accept.** `linux_x86_64` is
  what `sysconfig.get_platform()` reports and it is not a tag Warehouse takes;
  Linux wheels must be `manylinux*` or `musllinux*`, which also carry the glibc
  floor that says which systems the binary actually runs on.

* **An sdist that cannot run its own test suite.** A packager or auditor who
  downloads the tarball to decide whether to trust it is the reader this
  project most wants to satisfy, and the suite is the argument.

Exits non-zero and says which artefact and why. Nothing here needs network
access or an install.
"""

from __future__ import annotations

import sys
import tarfile
import zipfile
from pathlib import Path

# Warehouse accepts these platform tag families and rejects everything else.
ACCEPTED_PLATFORMS = ("manylinux", "musllinux", "macosx", "win", "any")

# What the sdist has to carry for `pytest` to collect and pass inside it.
SDIST_NEEDS = ("tests/", "conformance/", ".githooks/", "src/plumbline/",
               "native/", "README.md", "LICENSE", "NOTICE")


def check_wheel(path: Path) -> list[str]:
    problems = []
    tag = path.stem.split("-")[-1]

    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
    core = [n for n in names if "/_plumbline." in n or n.startswith("_plumbline.")]

    if tag == "any":
        problems.append(
            "tagged py3-none-any, so it carries no compiled core and would be "
            "served to every platform in preference to a real wheel. Build the "
            "wheel directly rather than from the sdist:\n"
            "        python native/build.py\n"
            "        python -m build --wheel --no-isolation")
    elif not core:
        problems.append(f"tagged {tag} but contains no _plumbline binary")

    if tag != "any" and not tag.startswith(ACCEPTED_PLATFORMS):
        problems.append(
            f"platform tag {tag!r} is not one PyPI accepts "
            f"({', '.join(ACCEPTED_PLATFORMS)}). A Linux wheel built on the "
            "runner comes out `linux_x86_64`, which upload rejects and which "
            "records no glibc floor; repair it with `auditwheel repair`.")
    return problems


def check_sdist(path: Path) -> list[str]:
    with tarfile.open(path) as archive:
        names = archive.getnames()
    root = names[0].split("/")[0] if names else ""
    inside = {n[len(root) + 1:] for n in names if n.startswith(root + "/")}

    missing = [need for need in SDIST_NEEDS
               if not any(n == need.rstrip("/") or n.startswith(need)
                          for n in inside)]
    if missing:
        return [f"missing {', '.join(missing)} — the test suite shipped in it "
                "cannot collect, so nobody can check this tarball by running it"]
    return []


def main(argv: list[str]) -> int:
    targets = [Path(a) for a in argv[1:]] or [Path("dist")]
    files: list[Path] = []
    for target in targets:
        files.extend(sorted(target.glob("*")) if target.is_dir() else [target])

    wheels = [f for f in files if f.suffix == ".whl"]
    sdists = [f for f in files if f.name.endswith(".tar.gz")]
    if not wheels and not sdists:
        print(f"No distributions found in {', '.join(map(str, targets))}.",
              file=sys.stderr)
        return 2

    failures = 0
    for wheel in wheels:
        for problem in check_wheel(wheel):
            print(f"  {wheel.name}\n      {problem}", file=sys.stderr)
            failures += 1
    for sdist in sdists:
        for problem in check_sdist(sdist):
            print(f"  {sdist.name}\n      {problem}", file=sys.stderr)
            failures += 1

    if failures:
        print(f"\n{failures} problem(s). Not fit to publish.\n", file=sys.stderr)
        return 1

    print(f"{len(wheels)} wheel(s) and {len(sdists)} sdist(s) fit to publish.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
