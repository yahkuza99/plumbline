"""All three decoders must refuse the same frames, whether or not they run.

The README, the package docstring and the changelog all say this. It was not
true: `turbo` validated only the component count, so it accepted subsampled
frames that `reference` and `native` rejected. The check existed in three
places — twice by copy-paste and once not at all — which is how a rule drifts.

Two of these tests do not decode anything. They read the source, because the
machine that runs them may not be able to import numba, and a guarantee that
holds only where a decoder happens to be installed is not a guarantee. The
first version of this fix shipped a `turbo` that called `scan_slots` without
importing it, and every test still passed, for exactly that reason.
"""

import ast
from pathlib import Path

import pytest

from plumbline import native, reference

SOURCE = Path(reference.__file__).parent
ENGINES = ("reference", "native", "turbo")


def _module(name):
    return ast.parse((SOURCE / f"{name}.py").read_text(encoding="utf-8"))


def _imported_from_reference(tree):
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "plumbline.reference":
            for alias in node.names:
                names.add(alias.asname or alias.name)
    return names


@pytest.mark.parametrize("engine", ("native", "turbo"))
def test_shared_checks_an_engine_calls_are_actually_imported(engine):
    """A decoder that cannot be imported here is a decoder nothing tests.

    `turbo` needs numba, which does not install against every NumPy. Where it
    is absent the module is never loaded, so a name it uses without importing
    is invisible to the entire suite until a user with a working numba runs
    into it. That is not hypothetical: the commit that shared these checks
    shipped a `turbo` calling `scan_slots` it had never imported, and every
    test still passed.
    """
    tree = _module(engine)
    imported = _imported_from_reference(tree)
    shared = {"check_frame", "check_table", "check_scan", "scan_slots",
              "header", "LosslessJpegError", "MAX_PRECISION"}

    used = {node.func.id for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    used |= {node.id for node in ast.walk(tree)
             if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}

    missing = (used & shared) - imported - {"_slots"}
    assert not missing, (
        f"{engine}.py uses {sorted(missing)} from the shared checks without "
        "importing them — it will raise NameError the moment it runs")


@pytest.mark.parametrize("engine", ENGINES)
def test_every_engine_validates_through_the_shared_check(engine):
    """The refusals must come from one place, not three copies of it."""
    source = (SOURCE / f"{engine}.py").read_text(encoding="utf-8")
    assert "scan_slots" in source, (
        f"{engine}.py does not go through scan_slots, so its refusals can "
        "drift from the other two")
    if engine != "reference":
        assert "sampling factor" not in source, (
            f"{engine}.py has its own copy of the subsampling check; it should "
            "import the shared one")


def test_the_engines_that_run_here_refuse_a_subsampled_frame():
    """And the guarantee holds in fact, not only in the source."""
    from conformance import frames as build

    sof = bytes([8, 0, 2, 0, 2, 1, 0, 0x22, 0])          # 2x2 sampling
    frame = (build.marker(0xD8) + build.marker(0xC3, sof)
             + build.huffman_table([1] + [0] * 15, [0])
             + build.marker(0xDA, bytes([1, 0, 0x00, 1, 0, 0]))
             + b"\x00\x10" + build.marker(0xD9))

    decoders = [("reference", reference.decode)]
    if native.AVAILABLE:
        decoders.append(("native", native.decode))
    try:
        from plumbline import turbo
        if turbo.AVAILABLE:
            decoders.append(("turbo", turbo.decode))
    except Exception:
        pass

    for name, decode in decoders:
        with pytest.raises(reference.LosslessJpegError):
            decode(frame)
