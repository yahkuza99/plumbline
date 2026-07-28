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
  refused — never retried down the chain. All three now refuse the same
  frames for the same reasons: `reference.check_frame`, `check_table` and
  `check_scan` are imported by the other two rather than reimplemented.
- `LosslessJpegError` for everything not implemented, rather than a best
  effort: subsampling, truncated scans, missing tables, malformed markers.
- Six more families of malformed frame are refused instead of decoded into
  plausible pixels. Each was a way to invent an image out of bytes the
  encoder did not write:
  - a frame zero samples wide, or with no lines (T.81 §B.2.2 gives X the
    range 1..65535, and Y = 0 defers the height to a DNL marker this decoder
    does not implement) — previously returned an empty array;
  - an over-subscribed Huffman table, declaring more codes of a length than
    that length has room for — previously lost whichever symbols ran off the
    end, silently and differently in each decoder;
  - a scan carrying a code its own Huffman table never defines — previously
    consumed no bits and called the difference zero, which produces a whole
    image from a stream that cannot be read;
  - a declared restart interval with fewer RST markers than it needs;
  - restart markers out of the RST0-RST7 cycle, which is how a decoder is
    meant to notice that markers were lost;
  - a 0xFF inside the entropy-coded data followed by neither byte stuffing
    nor a marker.
- **OUTPUT CHANGE** for those frames only, and from pixels to a refusal.
  No frame that decoded correctly before decodes differently now: the 1,691
  conforming frames of the conformance corpus and all 11 real discs are
  unchanged, bit for bit.
- The reference decoder now refuses a truncated scan, as `turbo` and `native`
  already did, instead of reading into its own padding and returning an image
  whose tail it invented.

### Speed

Median 117.9 Mpx/s over 11 frames from real discs (8 manufacturers), one AMD
Zen 3 core — about 25× `pylibjpeg-libjpeg` 2.4.0 on the same frames, which
decoded all 11 of them correctly. Reproduce with `native/compare.py`.

Validating the entropy-coded segment costs about 9% of that, measured on the
same frames (135.5 → 123.9 Mpx/s median, best-of-seven): it reads every byte
of the scan a second time to find the 0xFF markers, which `native`'s C
destuffing already walks and could report for free. Writing that logic twice
in two languages is how the two copies start to disagree, which is the failure
this project exists to prevent, so the second pass stays until the speed
matters more than that. See `reference.check_scan`.

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
