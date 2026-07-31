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

A build with no core present stays pure `py3-none-any`. That is a real and
supported outcome — it is what `pip install` does from an sdist on a platform
with no wheel — so it cannot be made an error here. It is also exactly what
`python -m build` produces by accident, because that command builds the wheel
*from the sdist it just made*, and an sdist can never carry a compiled binary.
The two are indistinguishable at this point in the build, so the guard against
publishing one lives where it can tell them apart: `tools/check_dist.py`,
which reads the finished artefacts.
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
                "no compiled core found — building a pure-Python wheel. This "
                "is correct for a source install and wrong for a release; "
                "tools/check_dist.py is what tells the two apart."
            )
            return

        platform = sysconfig.get_platform().replace("-", "_").replace(".", "_")
        build_data["pure_python"] = False
        build_data["tag"] = f"py3-none-{platform}"
