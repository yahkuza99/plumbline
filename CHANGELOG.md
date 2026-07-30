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

### Two misreadings of Annex H, found before the first release

Both were in all three decoders at once, because all three were written from
the same reading. Neither could have been caught by the real-frame corpus:
every one of its 61,921 frames uses predictor 1, which makes the first change
a no-op, and the second lived in the conformance *encoder* rather than in the
decoders. Both were found by reading T.81 Annex H sentence by sentence and
checking each against libjpeg, which shares no code with this project.

- **OUTPUT CHANGE — the first line of every restart interval now predicts
  from Ra.** T.81 §H.1.2.1: *"The one-dimensional horizontal predictor
  (prediction sample Ra) is used for the first line of samples at the start of
  the scan and at the beginning of each restart interval."* Plumbline applied
  the initial prediction *value* to the one sample after the marker — which
  the same paragraph also requires — and then let the rest of that line use
  the predictor from the scan header. That is one of the two rules, not both.

  Affects frames using **predictors 2–7 together with restart markers**, and
  nothing else: under predictor 1 the selected predictor *is* Ra, so the two
  readings compute the same image. All 11 real discs and every predictor-1
  frame are unchanged, bit for bit. Against libjpeg on a sweep of 840
  restart-carrying frames across predictors 2–7, agreement went from 320/840
  to 840/840.

- **OUTPUT CHANGE — a restart interval that is not a whole number of MCU-rows
  is now refused.** T.81 §H.1.1: *"For the lossless processes the restart
  interval shall be an integer multiple of the number of MCU in an MCU-row"*,
  with table B.7 giving Ri for lossless as `n × MCUR`. Such a frame is
  non-conforming, and mid-row the rule above has no "first line" to name:
  three readings are defensible, and libjpeg reproduces none of them. Pixels
  that only this decoder computes are the failure this project exists to
  prevent, so it refuses instead. Of 2,501 real frames sampled for this,
  every one that restarts at all restarts exactly once per row; none is
  affected.

- **The conformance encoder took differences modulo 2^P instead of modulo
  2^16** (§H.1.2.1: *"The difference between the prediction value and the
  input is calculated modulo 2^16"*). The decoders were unaffected — masking
  with 2^P is the same number for any conforming frame — but the corpus was
  not: it published frames whose differences were in the wrong residue class
  mod 2^16, which only decoders sharing the mistake could read. Against the
  1,707-case corpus, pylibjpeg went from **234 exact / 1,472 wrong** to
  **1,690 exact / 17 wrong**, and the 17 that remain are pylibjpeg's own gaps
  — 11 point-transform cases it does not implement and 6 malformed frames it
  accepts.

  The lesson is the one the corpus README already stated and this project
  still managed to relearn: a round trip through your own encoder proves only
  that your two halves share assumptions. It is not evidence until something
  that shares no code with you agrees.

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

- 61,921 lossless-JPEG frames from real files (26.1 billion pixels, 105 scanner
  models): **0 refused, 0 crashed**. One frame from each of the 137 parameter
  combinations re-decoded with the pure-Python reference: **185 exact, 0
  differing**.
- 11/11 real discs bit-exact at full size, every frame · colour bit-exact ·
  a 1,707-case conformance corpus over table shape × precision × predictor ×
  components × restart interval × point transform, all **1,707 exact, 0
  wrong** · **zero silent disagreements** · 184 tests passing, 2 skipped, on a machine without numba.
- Our three implementations are verified against the pure-Python reference in
  this repository. The conformance corpus is additionally run against
  decoders that share no code with us, because agreement among three
  implementations written from one reading of the specification is not
  evidence that the reading was right — as the section above records, twice.

### Not carried over

No git history from the application this came out of. That repository
contains patient images in its history and will never be made public; this
one starts clean, with a pre-commit hook and a CI job that both refuse
medical images by magic bytes rather than by filename.
