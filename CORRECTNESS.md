# What this project claims, and what backs the claim

The reason Plumbline exists is not speed. It is that a decoder can be wrong
without anybody noticing, and in medical imaging that is the worst possible
failure: an image that looks like an image, is displayed like an image, and
is not the image that was recorded.

So the claims below are narrow on purpose, and each one names the evidence.

## The rule

> **Decode correctly, or raise. Never return plausible wrong pixels.**

Every design decision in this project is subordinate to that rule, including
speed and including coverage. Where a frame uses something not implemented —
subsampling, a truncated scan, a missing table — the answer is
`LosslessJpegError`, not a best effort.

This is testable, and it is tested: the suite asserts that refusing is always
permitted, and that *returning pixels the reference implementation would not
produce* never is.

## What is implemented

| | |
|---|---|
| Format | ITU-T T.81 Annex H — lossless JPEG, SOF₃ |
| DICOM transfer syntaxes | `1.2.840.10008.1.2.4.57` (process 14) · `1.2.840.10008.1.2.4.70` (process 14, selection value 1) |
| Precision | 2–16 bits |
| Predictors | 1–7 |
| Components | 1 (greyscale) and interleaved multi-component (colour) |
| Restart markers | yes (DRI / RST₀–RST₇) |
| Point transform | yes |

**Not implemented, and refused rather than guessed:** encoding · chroma
subsampling · JPEG-LS (`.80`/`.81`) · JPEG 2000 (`.90`/`.91`) · baseline JPEG
(`.50`) · hierarchical and arithmetic-coded variants · anything to do with
parsing DICOM itself. This library takes a JPEG frame and returns pixels.
Use pydicom or GDCM for the container.

## The three traps

Most of the correctness work went into three places in the specification
where a decoder can look finished and be wrong. They were each found by
decoding real discs and diffing, not by reading the spec harder:

1. **Restart markers reset the predictor and re-align to a byte boundary.**
   A decoder that treats `0xDD`/`RSTn` as ordinary data drifts from the first
   marker onward. Most real files carry one marker per row, so the corruption
   is subtle and structured — it looks like noise, not like a bug.

2. **SSSS = 16 is a special case** (T.81 H.1.2.2). The difference is 32768
   and **no bits follow**. Reading 16 bits here consumes the next symbol and
   desynchronises the rest of the scan.

3. **Reconstruction wraps modulo 2^P**, and this is a matter of correctness,
   not of speed: predictors 5–7 average neighbouring samples, so a value that
   wrapped must stay wrapped for the following predictions to be right.

## The evidence

| Check | Result |
|---|---|
| **Frames from real hospital discs** | **61,921 decoded · 0 refused · 0 crashed** (>10 billion pixels) |
| Distinct scanner models in that corpus | **93** |
| Distinct parameter combinations found | 117 (precision × predictor × components × restart × model) |
| Covering sample re-decoded with the pure-Python reference | **88 exact, 0 differing** |
| Colour frames | bit-exact against the reference implementation |
| Synthetic conformance sweep | table shape × precision × predictor × restart interval × point transform |
| Silent disagreements between implementations | **0** |
| Test suite | 261 tests |

Comparing every frame against the reference is not possible — it runs at
about 0.15 Mpx/s, so ten billion pixels would take roughly a day. Instead,
every frame is decoded by the shipping decoder (which catches refusals,
crashes and hangs), and one frame from every distinct parameter combination
is then re-decoded with the reference and compared bit for bit.

### Where that corpus came from, and what it does not cover

**Every disc is from a hospital in Thailand.** That is a real limitation and
not a small one: the 93 models are the machines Thai hospitals bought, and
the distribution is dominated by Siemens, GE and Philips CT and MR. Scanners
common in other markets may not appear even once.

Concretely, the corpus is **thin or silent** on: Canon Medical / Toshiba,
Konica Minolta, Carestream, Shimadzu, Mindray, Samsung; machines older than
roughly 2010; and anything re-encoded by a vendor's own export or archiving
software rather than written by the scanner.

Every predictor observed in it was **1**. Predictors 2–7 are implemented and
covered by the synthetic suite, but no real disc here has exercised them, so
they carry less evidence than the numbers above might suggest.

If you have discs from elsewhere, running `native/compare.py` against them is
the single most useful thing you can do for this project — no files need to
leave your machine, and a disagreement is worth more to us than a patch.

Everything is verified against `plumbline.reference` — the pure-Python
implementation in this repository — and never against another library. That
is deliberate. Testing a decoder against a second decoder tests only that
they share assumptions; the reference is written to be read against the
specification clause by clause, and its comments cite those clauses.

Reproduce it with:

    python native/build.py
    python -m pytest tests/ -q

Against your own discs (nothing is uploaded, nothing leaves your machine):

    PLUMBLINE_TESTDATA=/path/to/your/files PLUMBLINE_FULL_TESTDATA=1 python -m pytest tests/ -q

## What this does *not* claim

- **Not a medical device.** Not submitted to any regulator, not for
  diagnostic use. If you put it in a regulated product, its verification is
  your responsibility — this document exists to make your job easier, not to
  do it for you.
- **Not exhaustively proven.** "No known silent failures on the corpus we
  have" is a much weaker statement than "correct", and it is the strongest
  one anybody can honestly make about a decoder. If you find a frame this
  library gets wrong, that is the most valuable thing you can send us.
- **Not a complete DICOM library.** See the "not implemented" list above.

## If you find a wrong pixel

Open an issue with the frame that reproduces it. **Strip the patient data
first** — the pixel data and the JPEG headers are all that is needed, and the
issue template will not accept anything with a DICOM preamble attached.

A frame that decodes wrongly is worth more to this project than a pull
request. It is the one thing we cannot generate ourselves.
