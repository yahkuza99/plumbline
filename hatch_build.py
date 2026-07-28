"""Tag the wheel for the platform whose decoder core it carries.

Without this the wheel comes out `py3-none-any`, and pip would happily
install a Windows `.dll` on a Linux box. Plumbline would still work there —
it falls back — but the user would have downloaded a useless binary and the
wheel would be lying about what is inside it.

The tag is `py3-none-<platform>` rather than `cp312-cp312-<platform>`: the
core is plain C that never touches the Python C API, so one binary is valid
for every Python version on that platform. That is the whole reason for
using ctypes instead of a extension module, and it means three wheels per
release instead of three per release per supported interpreter.

A build with no core present stays pure `py3-none-any` — that is the
source-only install, which falls back to numba or to the reference
implementation.
"""

import sysconfig
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

CORE = "_plumbline"


class PlatformTagHook(BuildHookInterface):
    PLUGIN_NAME = "custom"

    def initialize(self, version, build_data):
        package = Path(self.root) / "src" / "plumbline"
        built = [p for p in package.glob(CORE + ".*")
                 if p.suffix in (".dll", ".so", ".dylib")]

        if not built:
            self.app.display_warning(
                "no compiled core found — building a pure-Python wheel. "
                "Run `python native/build.py` first to ship the fast path."
            )
            return

        platform = sysconfig.get_platform().replace("-", "_").replace(".", "_")
        build_data["pure_python"] = False
        build_data["tag"] = f"py3-none-{platform}"
