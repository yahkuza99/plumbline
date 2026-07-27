# Contributing

## Before anything else: no patient data

The most valuable thing you can send this project is **a frame it decodes
wrongly**. It is also the most dangerous, because those frames come out of
real scanners.

Strip it first. The JPEG frame alone is what is needed — not the DICOM file:

```python
import pydicom
from pydicom.encaps import generate_fragmented_frames

ds = pydicom.dcmread("yours.dcm")
frame = b"".join(next(iter(generate_fragmented_frames(ds.PixelData))))
open("frame.jpg", "wb").write(frame)      # pixels only, no patient tags
```

A pre-commit hook refuses anything that looks like a medical image. Turn it on
once, in your clone:

```sh
git config core.hooksPath .githooks
```

It reads magic bytes, not filenames, so renaming a file does not get past it.
CI runs the same check over the whole history on every push.

## Build and test

```sh
python native/build.py          # finds cl, gcc, clang or zig cc
python -m pytest tests/ -q
```

Against your own discs:

```sh
PLUMBLINE_TESTDATA=/path/to/discs PLUMBLINE_FULL_TESTDATA=1 python -m pytest tests/ -q
```

**If `pytest` passes, you have not broken anything. Send the PR.**

## What gets merged

**Accepted without discussion**
- a bug fix with a test that fails before it and passes after
- a synthetic frame that reproduces a decoding error
- a build fix for a platform we have not tested
- documentation and typo fixes

**Open an issue first**
- new features, public API changes, a new dependency
- anything that changes decoded output — even by one bit

**Declined**
- reformatting or refactoring that does not fix a defect
- widening the scope (see the "does not" list in the README)
- changes without a test

The scope is closed on purpose. It is the only reason one maintainer can make
a correctness promise about this library at all, so proposals to widen it get
a polite no — usually with a suggestion of where the feature does belong.

## Style

Match the file you are editing. Comments explain *why*, and where they touch
the format they cite the clause — `// ITU-T T.81 §H.1.2.2` — so the next
person can check the code against the specification rather than against
their assumptions.

There is no formatter to run and no style bot. If something needs
reformatting, that is a review comment, not a merge blocker.

## The one rule that outranks everything

> **Decode correctly, or raise. Never return plausible wrong pixels.**

A change that makes the decoder faster, or that accepts more files, and which
can produce a wrong image without raising, will not be merged no matter how
much faster or how many more files. If you are unsure whether a change can do
that, say so in the PR and we will work it out together — that is a good
question, not an admission of anything.

## Legal

Contributions are under Apache-2.0, the project's licence. Sign off your
commits (`git commit -s`) to certify you have the right to submit them — the
[Developer Certificate of Origin](https://developercertificate.org/).

**There is no CLA.** You keep the copyright in what you write. The consequence
is that this project can never be relicensed without asking every contributor,
and that is intended: it is the guarantee that what you contribute to a free
library stays in one.
