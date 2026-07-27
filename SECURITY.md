# Security

## Reporting

Use GitHub's [private vulnerability reporting](../../security/advisories/new)
on this repository. Do not open a public issue.

Expect an acknowledgement within a week. If you have heard nothing in two
weeks, open a public issue saying only that you are waiting on a private
report — no details — and that will get attention.

## What counts

Plumbline parses untrusted binary data from files of unknown origin, and part
of it is C. That is exactly the shape of a memory-safety problem, so these
are all in scope:

- a frame that causes an out-of-bounds read or write, a crash, or an
  unbounded allocation in the compiled core
- **a frame that decodes to wrong pixels without raising** — silent data
  corruption is treated as a security issue here, not merely a bug, because
  the harm is that nobody can tell
- a frame that makes the decoder loop or allocate without limit

Out of scope: bugs in the container parser you used to extract the frame
(that is pydicom or GDCM, not this project), and anything requiring the
attacker to already be able to run code on the machine.

## What we can promise

One maintainer, no bug bounty, no cash. What you get instead: credit in the
advisory and the changelog under whatever name you choose, a fix released as
soon as it is written, and a straight answer about severity — including
"this is not exploitable and here is why".

**No paid bug bounty, deliberately.** Other single-maintainer projects have
been drowned by generated reports chasing rewards. A report that is real gets
taken seriously here regardless of whether money is attached.

## Attachments

Send the JPEG frame, never the DICOM file. If a vulnerability only reproduces
with a real clinical image, say so and we will work out how to reproduce it
synthetically — **do not attach patient data to a security report.** An
advisory is published; the attachment would be too.
