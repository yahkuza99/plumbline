# Conformance corpus — not ready to publish

The generator here is incomplete and the corpus it produced has been deleted
rather than committed. Read this before regenerating it.

## What went wrong

`generate.py` produced 206 synthetic frames and asserted the pixels each one
must decode to. Run against other decoders, 114 of the 206 disagreed with us
and **no independent decoder agreed with us on any of them**.

That number is the alarm. A suite in which the majority of cases declare
every other implementation wrong is far more likely to be broken itself, and
it was:

> **Frames with a restart interval carried the `DRI` marker in the header but
> contained no `RSTn` markers in the entropy data at all.**

The scan was pseudo-random bytes. A real encoder ends each restart interval
by padding to a byte boundary and emitting `RST0`–`RST7` in sequence; random
bytes contain none. So those frames were malformed, and every decoder that
disagreed with us was reacting to a broken input in its own reasonable way.

The evidence that pinned it: on **real** discs carrying restart markers,
pylibjpeg agreed with our reference bit for bit on all of them. A decoder
that handles restart markers correctly on real files and "fails" on ours is
telling us about our files.

## Why this cannot be patched

You cannot build a valid restart-marker stream out of random bytes. The
decoder consumes an unpredictable number of bits per interval, so it never
arrives at the byte boundary where the marker was placed. The interval
boundary has to be produced by whatever emitted the bits.

## The fix, and why it makes the corpus better

Write a **minimal encoder for the generator**: choose the image, emit its
differences as Huffman codes, pad to a byte boundary at each interval, emit
the right `RSTn`. Two things follow.

1. Every frame becomes well-formed, so a disagreement means something.
2. **The expected pixels are the image we started from**, not "whatever our
   own reference produced". That removes the circularity: today the corpus
   asserts one implementation's reading of the specification; afterwards it
   asserts the round trip, which is a fact rather than an opinion.

To be clear about scope: that encoder belongs to the test generator and is
never part of the shipped library. Plumbline decodes and will not write
files — a decoder that is wrong is visible immediately, an encoder that is
wrong corrupts an archive silently.

## Do not regenerate and commit until

- [ ] intervals end with a real, byte-aligned `RSTn`, cycling `RST0`–`RST7`
- [ ] expected pixels come from the source image, not from a decoder
- [ ] every case has at least one independent decoder agreeing, **or** a
      written argument citing the T.81 clause that says why it should not
