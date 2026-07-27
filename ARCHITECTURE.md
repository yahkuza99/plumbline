# How this is put together

Read this before changing anything. It is short because the project is small,
and the project is small on purpose.

## The shape

```
        bytes of one JPEG frame
                  │
        plumbline.decode()            src/plumbline/__init__.py
                  │                   picks by availability, never retries
     ┌────────────┼────────────┐
     ▼            ▼            ▼
  native.py    turbo.py   reference.py
  ctypes ──►   numba        pure Python
  _plumbline   greyscale    THE ORACLE
  .dll/.so     only         everything is tested against this
     │
     ▼
  native/plumbline.c        the only C in the project
```

Every implementation must produce **exactly** what `reference.py` produces,
bit for bit, or refuse. That is the entire contract between them, and it is
what the test suite checks on every frame it can find.

## The files, and which ones you will actually touch

| File | Lines | What it is |
|---|---|---|
| `src/plumbline/reference.py` | ~310 | **Start here.** Pure Python, no cleverness, comments cite T.81 clause numbers. It is the specification made runnable. |
| `native/plumbline.c` | ~690 | The fast path. Where nearly all bug fixes land. |
| `src/plumbline/native.py` | ~200 | Parses the JPEG headers in Python, validates, then hands the scan to C. **The C never parses headers** — that keeps the attack surface in the memory-unsafe language as small as it can be. |
| `src/plumbline/turbo.py` | ~440 | numba fallback. Greyscale only. Kept for platforms with no wheel. |
| `src/plumbline/__init__.py` | ~75 | Picks an implementation. Public API. |
| `native/build.py` | ~100 | Finds a compiler (`cl` → `gcc` → `clang` → `zig cc`) and builds the core. |
| `native/compare.py` | ~180 | Benchmarks *and diffs* against every other decoder installed. |
| `native/verify.py` | ~75 | Decodes every frame of every disc in the corpus, twice, and diffs. |

## Why C and ctypes, rather than a Python extension

The core exports two plain C functions and touches no Python C API at all.
That buys three things:

- **one binary per platform, valid for every Python version** — a
  `py3-none-<platform>` wheel rather than one per interpreter version
- **any compiler will do**, so CI uses each runner's system toolchain and
  nothing has to be downloaded to build it
- **a hard boundary**: everything about JPEG headers, tables and validation
  is decided in Python. The C receives an already-validated scan, a table,
  and an output buffer it is told the size of.

The cost is an ABI to keep in step. `plumbline_abi()` returns a version that
`native.py` checks on load; if they disagree the library is ignored and the
fallback takes over, which is why a stale `.dll` can never silently produce
wrong pixels.

## Where the speed comes from

Measured, in the order the wins were found — every one of them kept only
because a benchmark said so, and one was reverted for the same reason:

1. **A 16 KB fused Huffman table that fits in L1.** The obvious 256 KB table
   sat in L2, and the lookup is on a serial dependency chain, so every miss
   was exposed latency. This one change was worth about 40%.
2. **Peeling the first row and first column out of the interior loop**, so
   the predictor is a compile-time constant everywhere else.
3. **Interleaving four restart intervals in one core** for instruction-level
   parallelism — not threads. Gated to intervals of exactly one row: on a
   mammogram with 25-row intervals it measured *slower* (52 vs 76 Mpx/s), so
   it is off there.

Decoding is inherently serial — each pixel is predicted from its neighbours —
so the wins are all about hiding latency, never about doing less work.

## The rule that constrains every change

> **Decode correctly, or raise. Never return plausible wrong pixels.**

Concretely, when editing the C: a bounds check that costs 2% stays. A frame
that runs off the end of its scan returns an error code, and `native.py`
turns it into `LosslessJpegError`. The reference implementation is the one
place that *does* read into its own padding — which is exactly why it is the
oracle for correctness and never the decoder anyone ships against untrusted
input.
