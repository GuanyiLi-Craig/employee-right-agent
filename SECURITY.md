# Security policy

## Reporting a vulnerability

Please do not open a public issue.

Report privately through GitHub's
[private vulnerability reporting](https://github.com/GuanyiLi-Craig/employee-right-agent/security/advisories/new)
on this repository. Include what you did, what you expected, what happened, and
the commit you were on. If you have a nuclei template or a curl line that
reproduces it, that is ideal — see `security/nuclei/` for the house style.

Expect an acknowledgement within a few days. This is a demonstration project
maintained in spare time, so please size your expectations accordingly.

## What is in scope

This is a **demonstration application**, not a production service, and its
threat model is written down rather than implied. Read
[`security/README.md`](security/README.md) first: it records what was scanned,
what was fixed, and what is accepted risk with the reasoning attached.

Findings that are in scope:

- The dashboard's authorisation model: `RIGHTS_DEMO_TOKEN`, which endpoints it
  gates, and any path around it.
- The audit record: anything that lets the hash chain be broken, rewritten, or
  silently truncated without `verify_records` noticing.
- Session isolation: a way to read a transcript you should not have, other than
  by holding its id.
- Redaction: personal data reaching the audit record or the trace store that
  should have been redacted.
- The ingest pipeline: a corpus that produces a plausible but wrong tree, or a
  half-published index.

## What is out of scope, and why

These are **documented design decisions**, not oversights. `security/README.md`
gives the full reasoning for each.

- **Everything is unauthenticated when no `RIGHTS_DEMO_TOKEN` is set and the
  server is on loopback.** That is the local demo posture and it is deliberate.
  The server refuses to bind a non-loopback address with no token. Report
  findings against the posture you can actually reach on a published address.
- **The session id is the credential for its transcript.** Anyone holding one
  can read that conversation back. It is minted from a CSPRNG at 128 bits, and
  "I can read a transcript whose id I was given" is not a finding. A way to
  *guess* or enumerate one is.
- **`/api/chunks` is unauthenticated on purpose.** It reads public UK
  legislation that every answer already quotes, and changes no state.
- **The corpus is trusted.** Retrieved text enters the prompt, so a corpus an
  attacker can write to is an indirect prompt-injection channel. The corpus is
  committed to the repository and the index is built by a separate job, so the
  trust boundary is the repository itself.
- **No multi-tenant auth.** Explicitly out of scope for this project.

## Supported versions

The `main` branch only. There are no released versions and no backports.
