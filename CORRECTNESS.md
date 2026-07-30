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

## The traps

Places in the specification where a decoder can look finished and be wrong.
The first three were found by decoding real discs and diffing. The last two
could not have been: every real frame we hold uses predictor 1, which hides
them completely, and they were found only by reading Annex H clause by clause
and checking each sentence against an independent implementation.

1. **Restart markers reset the predictor and re-align to a byte boundary.**
   A decoder that treats `0xDD`/`RSTn` as ordinary data drifts from the first
   marker onward. Most real files carry one marker per row, so the corruption
   is subtle and structured — it looks like noise, not like a bug.

2. **SSSS = 16 is a special case** (T.81 H.1.2.2). The difference is 32768
   and **no bits follow**. Reading 16 bits here consumes the next symbol and
   desynchronises the rest of the scan.

3. **The wrap must happen inside the loop**, not over the finished image:
   predictors 5–7 average neighbouring samples, so a value that wrapped must
   stay wrapped for the following predictions to be right.

4. **A restart interval puts the whole first *line* back on Ra**, not just the
   sample after the marker. T.81 §H.1.2.1:

   > The one-dimensional horizontal predictor (prediction sample Ra) is used
   > for the first line of samples at the start of the scan and at the
   > beginning of each restart interval. The selected predictor is used for
   > all other lines. The sample from the line above (prediction sample Rb) is
   > used at the start of each line, except for the first line. At the
   > beginning of the first line and at the beginning of each restart interval
   > the prediction value of 2^(P – 1) is used, where P is the input precision.

   Two different rules live in that paragraph, and it is easy to read only the
   second. The prediction *value* 2^(P–1) applies at the **beginning** of an
   interval — one sample. The **Ra predictor** applies to the whole first
   **line** of it. Plumbline did the first and not the second until this was
   caught. Under predictor 1 the two readings compute the same number, which
   is why 61,921 real frames could not have found it: every one of them uses
   predictor 1.

   T.81 §H.1.1 is what makes the rule well defined: *"For the lossless
   processes the restart interval shall be an integer multiple of the number
   of MCU in an MCU-row"* — table B.7 gives Ri for lossless as `n × MCUR`. An
   interval therefore always begins a line. **An interval that is not a whole
   number of MCU-rows is refused**, because mid-row there is no "first line"
   for the clause to name; three readings are defensible and libjpeg
   reproduces none of them, so any pixels we returned would be ours alone.

5. **Differences are taken modulo 2^16, whatever the precision** (§H.1.2.1):

   > The difference between the prediction value and the input is calculated
   > modulo 2^16. In the decoder the difference is decoded and added, modulo
   > 2^16, to the prediction.

   **Not** modulo 2^P, which earlier versions of this document claimed. The
   two are the same number for every conforming frame — a sample is below 2^P,
   and 2^P divides 2^16, so the residue class has exactly one representative
   in range — and the decoders here mask with 2^P for that reason, which also
   keeps the result inside the precision the frame declares.

   Where it is *not* harmless is when writing: a difference reduced modulo 2^P
   is a different residue class mod 2^16 and decodes to a different sample
   everywhere but in the encoder that produced it. The conformance encoder did
   exactly that, and so published frames that only this project's decoders
   agreed with. Both halves shared the same misreading and round-tripped
   perfectly against each other, which is precisely why a corpus must be
   checked against a decoder that shares no code with it.

## The evidence

| Check | Result |
|---|---|
| **Lossless-JPEG frames from real files** | **61,921 decoded · 0 refused · 0 crashed** (26.1 billion pixels) |
| Distinct scanner models in that corpus | **93** |
| Distinct scanner builds found | 105 (manufacturer × model × modality), from 27 manufacturers |
| Covering sample re-decoded with the pure-Python reference | **185 exact, 0 differing** |
| Colour frames | bit-exact against the reference implementation |
| Synthetic conformance corpus | 1,707 cases — 1,691 must decode, 16 must refuse |
| …swept over | precision × predictor × components × restart interval × content × point transform |
| Plumbline against that corpus | **1,707 exact · 0 refused · 0 wrong** |
| pylibjpeg (shares no code with us) against it | **1,690 exact, 17 disagreements** — 6 malformed frames it accepts, 11 point-transform cases it does not implement |
| Silent disagreements between our own implementations | **0** |
| Test suite | 184 passing, 2 skipped, on a machine without numba; more where numba imports and the real discs are attached |

Comparing every frame against the reference is not possible — it runs at
about 0.15 Mpx/s, so 26 billion pixels would take about 48 hours. Instead,
every frame is decoded by the shipping decoder (which catches refusals,
crashes and hangs), and one frame from every distinct parameter combination
is then re-decoded with the reference and compared bit for bit.

### Where that corpus came from, and what it does not cover

**The vendor distribution is uneven**, and unevenly in a way that matters: the
105 configurations are dominated by Siemens, GE and Philips CT and MR, so a
machine common in another market may not appear even once. What did and did
not appear is listed in full below rather than summarised, because that is the
part a reader can check their own equipment against.

The 27 manufacturers that do appear are Siemens (and Siemens Healthineers /
Siemens NM), GE (MEDICAL SYSTEMS and Healthcare), Philips (and Philips Medical
Systems), **Canon Inc.** (CXDI), **Carestream Health** (DRX-1, DRX-Evolution,
DRX-Revolution), **FUJIFILM**, **Samsung** (Electronics GC85A / GM85, and
Medison V8), **HOLOGIC** (Discovery A / W / Wi, DXA), **GE Lunar Prodigy
Advance** (DXA), **TOSHIBA** (Aquilion ONE / PRIME, and TOSHIBA_MEC_US),
**MINDRAY** (Resona 7), Agfa, Hitachi (and Hitachi Aloka), DRTECH, E-COM,
EBM Technologies, Perceptra and Vital Images.

Earlier versions of this document listed Canon Medical / Toshiba, Carestream,
Mindray and Samsung as gaps. That was **wrong** — all four are in the table
above, and the claim is corrected here rather than quietly deleted.

What the corpus really is thin or silent on: **Konica Minolta** and
**Shimadzu**; machines older than roughly 2010; and anything re-encoded by a
vendor's own export or archiving software rather than written by the scanner.

**Every predictor observed in it was 1**, and this is the most important
limitation on the page. Predictors 2–7 are implemented and covered by the
synthetic corpus, but no real disc here has ever exercised them, so they carry
far less evidence than the frame count above might suggest. Both bugs fixed in
the release that added this paragraph were invisible under predictor 1 and
would have gone on being invisible for as long as the evidence was real frames
alone. The restart intervals are similarly narrow: of the frames that carry
markers at all, every one restarts exactly once per row.

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
