"""``python -m rights_agent <command>`` -- one entry point for the container.

Docker images get a single entrypoint and pick a subcommand, so the compose file
reads as a list of jobs rather than a list of module paths.
"""

from __future__ import annotations

import sys
from collections.abc import Callable

COMMANDS: dict[str, str] = {
    "ask": "rights_agent.cli:ask",
    "ingest": "rights_agent.pipelines.hierarchical:main",
    "ingest-simple": "rights_agent.pipelines.simple:main",
    "compare": "rights_agent.pipelines.compare:main",
    "demo": "rights_agent.demo.app:main",
    "corpus": "rights_agent.tools.corpus:main",
    "goldens": "rights_agent.tools.goldens:main",
    "evaluate": "rights_agent.tools.evaluate:main",
}


def _resolve(target: str) -> Callable[[list[str] | None], int]:
    module_name, _, attribute = target.partition(":")
    module = __import__(module_name, fromlist=[attribute])
    return getattr(module, attribute)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help", "help"}:
        width = max(len(name) for name in COMMANDS)
        print("usage: python -m rights_agent <command> [options]\n\ncommands:")
        for name, target in COMMANDS.items():
            print(f"  {name:<{width + 2}}{target}")
        return 0 if argv else 1
    command, *rest = argv
    if command not in COMMANDS:
        print(f"unknown command {command!r}; try --help", file=sys.stderr)
        return 2
    return _resolve(COMMANDS[command])(rest)


if __name__ == "__main__":
    raise SystemExit(main())
