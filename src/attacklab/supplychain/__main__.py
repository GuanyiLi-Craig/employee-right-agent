"""The supply-chain CLI.  Demo 3's ninety seconds, in one command.

    uv run python -m attacklab.supplychain --all

Prints the five scanners' findings in the order the demo strip expects, and
exits non-zero if anything failed -- so the same command is the CI gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from attacklab.attacks.corpus import clean_runs_dir, poison_runs_dir
from attacklab.supplychain import scan_index, scan_lock, scan_model, scan_skills, scan_tool_surface
from attacklab.supplychain.findings import ScanReport
from rights_agent.config import Settings
from rights_agent.config import settings as load_settings

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES = REPO_ROOT / "fixtures"


def run_all(settings: Settings | None = None) -> ScanReport:
    """Every scanner, in demo order."""
    settings = settings or load_settings()
    report = ScanReport()

    # 1. Same model, different container.
    report.extend(scan_model.scan(FIXTURES / "models", REPO_ROOT / "src"))

    # 2. The poisoned document, found by content. The manifest flag is never
    #    read -- see the module docstring and the test that enforces it.
    for label, runs in (("poisoned", poison_runs_dir(settings)), ("clean", clean_runs_dir(settings))):
        if not (runs / "chroma").exists():
            continue
        for finding in scan_index.scan(settings.with_overrides(runs_dir=runs)):
            report.add(
                type(finding)(
                    **{
                        **finding.to_dict(),
                        "subject": f"[{label} index] {finding.subject}",
                        "detail": finding.detail,
                    }
                )
            )

    # 3. A manifest requesting a tool it cannot justify.
    report.extend(scan_skills.scan(FIXTURES / "skills"))

    # 4. A moved digest.
    report.extend(scan_lock.scan(FIXTURES / "lock" / "models.lock.json", FIXTURES))

    # 5. The tool surface itself -- the check that comes before the others matter.
    report.extend(scan_tool_surface.scan())
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pre-deployment supply-chain scanners.")
    parser.add_argument("--all", action="store_true", help="run every scanner (the default)")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    report = run_all()
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.render())
    # Non-zero on failure, so `make gate` and the demo share one command.
    return 0 if report.ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
