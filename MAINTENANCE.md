# Maintenance

Written while the project is young and nobody is tired, because criteria
written after exhaustion are written out of guilt.

## What is promised

A first reply to an issue within about a week. A frame that reproduces a
decoding fault gets one sooner. Security reports go through
[SECURITY.md](SECURITY.md).

Nothing else. There is no roadmap, the scope in the README is closed, and the
maintainer has another job. That is stated so nobody has to guess at it.

## Review points

Checked every six months against these, and the outcome recorded here.

| At | If | Then |
|---|---|---|
| 6 months | nobody outside has installed it — no issues, no dependents | stop all promotion, stay in maintenance. Not a failure; the normal shape of infrastructure |
| 12 months | no user has reported running it against their own files | declare **stable, low activity** — not deprecated |
| any time | more than 4 hours a month for three months running, unpaid and unenjoyable | reduce scope immediately, without waiting for a review point |
| any time | a GitHub notification produces dread rather than interest | **stop.** This is the signal that preceded the xz compromise, and it costs more than the project is worth |

## Winding down, if it comes to that

In order of preference:

1. **Declare it finished, not abandoned.** ITU-T T.81 is a dead standard —
   there will be no new version — so a decoder with a closed scope can
   genuinely be complete. "Stable; correctness fixes and runtime compatibility
   only; 30-day response" is more useful to a downstream user than "actively
   developed", because it means no breaking changes are coming.
2. **Unmaintained but working.** State the last runtime it was tested against,
   and point at the alternatives honestly. Naming your competitors is what
   makes people trust the rest of the page.
3. **Hand it on — with conditions.** Commit rights go to someone who has
   landed three changes that passed the conformance suite. **Never to someone
   without a verifiable history, however helpful they seem, and least of all
   while tired.** That is precisely how xz-utils was compromised.
4. **Archive read-only. Never delete.**
   Not the repository, not the package, not any working release. Someone has
   pinned a version inside a validated pipeline, and in medical software
   changing a version means revalidating the whole thing.

If there are commercial users, give them **90 days' notice by email**. That
window is also, in practice, when someone offers to pay for it to continue.

## What is not a reason to stop

A quiet year. Few stars. No contributors. Infrastructure libraries in a narrow
field routinely have one author and no help for their entire life, and are
depended on by everything around them. Silence is the normal condition, not a
verdict.
