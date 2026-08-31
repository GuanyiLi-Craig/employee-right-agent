"""``rights-ask`` -- one question, with the numbers that say what it cost.

Prints the answer, its citations, and the measurements the specification asks
for: TTFT, ITL, end-to-end, the per-stage breakdown, tokens and cost.  A
refusal prints its score and the threshold it failed, because a refusal that
does not say why is indistinguishable from a bug.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from rights_agent.agent import Agent, AgentAnswer
from rights_agent.config import PRICING_AS_OF, price_for
from rights_agent.config import settings as load_settings
from rights_agent.embedding import EmbedderError
from rights_agent.entrypoints import operator_error_exit
from rights_agent.log import configure_logging
from rights_agent.store import IndexNotBuiltError, StoreError
from rights_agent.telemetry import shutdown_telemetry, telemetry_status

BAR_WIDTH = 44


def _bar(stage_ms: dict[str, float], width: int = BAR_WIDTH) -> list[str]:
    """A text stage bar, longest stage first."""
    total = sum(stage_ms.values()) or 1.0
    lines: list[str] = []
    for stage, value in sorted(stage_ms.items(), key=lambda item: -item[1]):
        filled = max(1, round(width * value / total))
        lines.append(f"  {stage:<10}{'█' * filled:<{width}} {value:>9.2f} ms")
    return lines


def render(answer: AgentAnswer, *, show_context: bool = False) -> str:
    metrics = answer.metrics
    price = price_for(metrics.model)
    lines: list[str] = []
    lines.append("")
    lines.append(answer.answer)
    lines.append("")
    if answer.citations:
        lines.append("citations   " + ", ".join(answer.citations))
    else:
        lines.append("citations   (none — refused)" if answer.refused else "citations   (none)")
    lines.append(
        f"gate        sufficiency {metrics.sufficiency:.3f} "
        f"(threshold {load_settings().sufficiency_threshold:.2f}, "
        f"attempts {metrics.attempts}, route {metrics.route})"
    )
    lines.append(
        f"retrieved   {len(metrics.retrieved_ids)} blocks — "
        + ", ".join(
            f"{citation}" for citation in [doc.get("citation", "") for doc in answer.docs]
        )
    )
    lines.append("")
    lines.append(
        f"latency     ttft {metrics.ttft_ms:.2f} ms · itl {metrics.itl_ms_mean:.2f} ms "
        f"(p95 {metrics.itl_ms_p95:.2f}) · e2e {metrics.e2e_ms:.2f} ms"
    )
    lines.append(
        f"            non-generation {metrics.non_generation_ms():.2f} ms · "
        f"orchestration {metrics.orchestration_ms():.2f} ms · "
        f"formula gap {metrics.formula_gap_ms():+.2f} ms"
    )
    lines += _bar(metrics.stage_ms)
    lines.append("")
    lines.append(
        f"tokens      prompt {metrics.prompt_tokens} "
        f"(cached {metrics.cached_tokens}) · completion {metrics.completion_tokens}"
    )
    components = " · ".join(f"{k} ${v:.6f}" for k, v in sorted(metrics.cost_breakdown.items()))
    lines.append(f"cost        ${metrics.cost_usd:.6f}  [{components}]")
    lines.append(f"model       {price.label(metrics.model)} · prices as of {PRICING_AS_OF}")
    if answer.scores:
        lines.append(
            "scores      "
            + " · ".join(f"{name} {value:.2f}" for name, value in sorted(answer.scores.items()))
        )
    lines.append(f"index       {metrics.index_version} · embedder {metrics.embedding_model}")
    lines.append(f"trace       {telemetry_status().status}")
    if metrics.trace_id:
        lines.append(f"            trace_id {metrics.trace_id}")
    if metrics.error:
        lines.append(f"error       {metrics.error}")
    if show_context:
        lines.append("")
        lines.append("context ----------------------------------------------------------------")
        lines.append(answer.context)
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rights-ask", description="Ask the indexed document a question."
    )
    parser.add_argument("question", nargs="+", help="the question")
    parser.add_argument("--session", default=None, help="session id (groups a conversation)")
    parser.add_argument("--user", default="", help="user id, recorded on the row and the span")
    parser.add_argument("-k", "--top-k", type=int, default=None, help="leaves to retrieve")
    parser.add_argument(
        "--threshold", type=float, default=None, help="sufficiency threshold for this call"
    )
    parser.add_argument(
        "--degraded",
        action="store_true",
        help="simulate a fallback model: lower-ranked evidence, no citations, slower",
    )
    parser.add_argument("--no-tracing", action="store_true", help="do not export spans")
    parser.add_argument("--show-context", action="store_true", help="print the assembled context")
    parser.add_argument("--json", action="store_true", help="emit the full record as JSON")
    parser.add_argument("--quiet", action="store_true", help="log warnings and errors only")
    return parser


@operator_error_exit
def ask(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    configure_logging("WARNING" if args.quiet else None)

    settings = load_settings()
    overrides: dict[str, object] = {}
    if args.top_k is not None:
        overrides["top_k"] = args.top_k
    if args.threshold is not None:
        overrides["sufficiency_threshold"] = args.threshold
    if args.no_tracing:
        overrides["tracing_enabled"] = False
    if args.degraded:
        overrides["degraded"] = True
    if overrides:
        settings = settings.with_overrides(**overrides)

    question = " ".join(args.question)
    try:
        agent = Agent(settings, degraded=args.degraded or None)
        answer = agent.ask(question, session_id=args.session, user_id=args.user)
    except (IndexNotBuiltError, StoreError, EmbedderError) as exc:
        # These are operator errors with a known fix, not bugs: print the fix
        # rather than a traceback.
        parser.exit(2, f"error: {exc}\n")
        return 2

    if args.json:
        print(json.dumps(answer.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(render(answer, show_context=args.show_context))
    shutdown_telemetry()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(ask())
