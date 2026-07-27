# Changelog

Anything that changes decoded output is marked **OUTPUT CHANGE**, even by one
bit, even when the new output is the correct one. If you are pinning this
library inside a validated pipeline, those are your re-validation triggers and
nothing else in here is.

## 0.1.0 — unreleased

First release. Extracted from the DICOM exporter it was written for, with no
history carried over.

### Decoder

- Lossless JPEG (ITU-T T.81 Annex H, SOF₃): precision 2–16, predictors 1–7,
  restart markers, point transform, greyscale and interleaved colour.
- Three implementations behind one entry point, chosen by what the machine
  has: compiled C, numba, pure Python. A frame the chosen decoder refuses is
  refused — never retried down the chain, because the reference would invent
  the tail of a truncated scan rather than notice it was truncated.
- `LosslessJpegError` for everything not implemented, rather than a best
  effort: subsampling, truncated scans, missing tables, malformed markers.

### Speed

Median 117.9 Mpx/s over 11 frames from real discs (8 manufacturers), one AMD
Zen 3 core — about 25× `pylibjpeg-libjpeg` 2.4.0 on the same frames, which
decoded all 11 of them correctly. Reproduce with `native/compare.py`.

The compiled core replaced numba as the default: it is faster, it starts in
0.26 s instead of 10, it decodes colour (numba does not), and it removes a
heavy dependency that lags each new Python release by months.

### Correctness

- 11/11 real discs bit-exact at full size, every frame · colour bit-exact ·
  synthetic sweep over table shape × precision × predictor × restart interval
  × point transform · **zero silent disagreements** · 261 tests.
- Everything is verified against the pure-Python reference in this
  repository, never against another library.

### Not carried over

No git history from the application this came out of. That repository
contains patient images in its history and will never be made public; this
one starts clean, with a pre-commit hook and a CI job that both refuse
medical images by magic bytes rather than by filename.
