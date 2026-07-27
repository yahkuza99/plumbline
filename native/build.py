"""Build the Plumbline shared library into src/plumbline/.

Usage:  python native/build.py [--cc <compiler>]

Tries, in order: the --cc argument, $PLUMBLINE_CC, cl (MSVC), gcc, clang, cc,
and finally a `zig` toolchain (PATH first, then the known local install),
using `zig cc` as the C compiler. The first one that exists is used; the
build is a single command over a single C file, so there is nothing else to
configure.

The output lands next to src/plumbline/native.py as `_plumbline.dll`,
`.dylib` or `.so`, which is where that module looks for it. CI wheels are the
same story on the runner's own compiler — the C is plain C11 with no
dependencies beyond libc.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = Path(__file__).resolve().parent / "plumbline.c"
SUFFIX = {"win32": ".dll", "darwin": ".dylib"}.get(sys.platform, ".so")
OUTPUT = ROOT / "src" / "plumbline" / ("_plumbline" + SUFFIX)

# A machine without any system compiler can drop a zig toolchain here and
# builds keep working; zig cc is a full clang.
KNOWN_ZIGS = [
    Path(os.environ.get("LOCALAPPDATA", "")) / "Programs"
    / "zig-x86_64-windows-0.16.0" / "zig.exe",
]


def _found(command: str) -> str | None:
    return shutil.which(command)


def _zig() -> str | None:
    if path := _found("zig"):
        return path
    for candidate in KNOWN_ZIGS:
        if candidate.is_file():
            return str(candidate)
    return None


def _command(cc: str, zig: bool) -> list[str]:
    if Path(cc).name.lower() in ("cl", "cl.exe") and sys.platform == "win32" and not zig:
        return [cc, "/nologo", "/O2", "/W3", "/LD", str(SOURCE),
                f"/Fe:{OUTPUT}"]
    front = [cc, "cc"] if zig else [cc]
    out = front + ["-O3", "-shared", "-o", str(OUTPUT), str(SOURCE),
                   "-Wall", "-Wextra"]
    if sys.platform != "win32":
        out.append("-fPIC")
    if zig:
        # zig cc traps undefined behaviour by default, which would abort the
        # host process instead of returning an error code.
        out.append("-fno-sanitize=undefined")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cc", help="compiler executable to use")
    arguments = parser.parse_args()

    tried: list[str] = []
    for cc, zig in ([(arguments.cc, False)] if arguments.cc else []) + \
                   ([(os.environ.get("PLUMBLINE_CC"), False)]
                    if os.environ.get("PLUMBLINE_CC") else []) + \
                   [(_found("cl"), False), (_found("gcc"), False),
                    (_found("clang"), False), (_found("cc"), False),
                    (_zig(), True)]:
        if not cc:
            continue
        tried.append(cc)
        command = _command(cc, zig)
        print("+", " ".join(command))
        completed = subprocess.run(command, cwd=ROOT)
        if completed.returncode == 0 and OUTPUT.is_file():
            for leftover in ("_plumbline.pdb", "plumbline.lib", "plumbline.exp",
                             "_plumbline.lib", "_plumbline.exp", "plumbline.obj"):
                (OUTPUT.parent / leftover).unlink(missing_ok=True)
                (ROOT / leftover).unlink(missing_ok=True)
            print(f"built {OUTPUT}")
            return 0
        print(f"{cc} failed with {completed.returncode}", file=sys.stderr)

    print("no C compiler found" if not tried else "every compiler tried failed",
          file=sys.stderr)
    print("install MSVC Build Tools, gcc, clang, or drop a zig toolchain at",
          KNOWN_ZIGS[0], file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
