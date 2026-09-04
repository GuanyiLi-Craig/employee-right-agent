"""Session 6 attack lab: attack the assistant, then defend it one control at a time.

A **defensive** lab.  Three rules constrain everything in it:

1. **No payload does real damage.** Injection payloads cause a wrong answer, or
   an *attempt* at a tool call that should not be allowed.  Nothing exfiltrates,
   nothing reaches the network, nothing outside the scratch directory is
   touched.  The point of every demonstration is the attempt and what happens
   to it, not the effect.
2. **No real malware.** :mod:`attacklab.supplychain` scans model files for
   dangerous deserialisation; the sample artefact's payload writes one marker
   file to a scratch path and nothing else.  The detector is the deliverable.
3. **Local and offline.** No live target other than the assistant on this
   machine, and the whole lab runs with no network and no API key.

The lab does not fork the assistant.  It installs itself at the seven hook
points in :mod:`rights_agent.hooks` and consults a registry of toggles on every
call, which is what makes the console's switches take effect on the *next*
request with no restart.
"""

# Deliberately no imports here. The console, the report CLI and the eval suite
# each import exactly the module they need, so a broken control cannot stop the
# supply-chain scanners from running -- on stage, a partial lab beats no lab.
__all__: list[str] = []
