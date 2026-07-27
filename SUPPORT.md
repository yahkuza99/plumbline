# Support

Plumbline is maintained by one person who has another job. That is not an
apology, it is the information you need to plan around — so here is what each
channel actually gets you.

| You have | Go to | You can expect |
|---|---|---|
| **A frame that decodes wrongly or is refused** | [Issues](../../issues) — attach the frame | A first reply within a few days. This is the report this project most wants. |
| A build failure on your platform | [Issues](../../issues) | A first reply within about a week |
| "How do I use this with X?" | [Discussions](../../discussions) | Answered when there is time. No promise. |
| "How does lossless JPEG / DICOM work?" | Not here | I am not a free consultancy. Start with ITU-T T.81 and the pydicom docs. |
| A security issue | [SECURITY.md](SECURITY.md) | Do not open a public issue |
| **A guaranteed response time, a fix on your schedule, a supported LTS branch, or evidence for a regulatory submission** | This is paid work — open an issue titled `commercial enquiry` and we will take it to email | — |

**Every attachment must be free of patient data.** Send the JPEG frame, not
the DICOM file. [CONTRIBUTING.md](CONTRIBUTING.md) has a five-line script that
extracts it.

## If you are putting this in a regulated product

[CORRECTNESS.md](CORRECTNESS.md) is written for you: what is implemented, what
is refused, the evidence behind each claim, and — importantly — what this
project does *not* claim. Under IEC 62304 this library is SOUP, and that
document is meant to cover most of what your file needs.

Two things worth knowing before you commit to it:

- Pinning a version is on you, and you should. Anything that changes decoded
  output gets its own entry in [CHANGELOG.md](CHANGELOG.md), even a one-bit
  change, so you can tell a re-validation trigger from a routine bump.
- If the maintainer disappears, you are not stranded. Apache-2.0, no CLA, no
  build step that depends on any service of ours, and a readable pure-Python
  reference implementation in the repository. Fork it.

## On reports written with AI assistance

They are fine — say so, and confirm you reproduced the problem yourself with
the output to show it. Reports that cannot be reproduced get closed without
discussion. This is not hostility toward the tools; it is that a single
maintainer cannot outpace a generator, and other projects have been buried
this way.
