"""Eval runner -- runs the REAL pipeline over evals/golden/ and prints a
pass-rate table. Exit code 0 when pass_rate >= --threshold, 1 when below,
2 on harness misconfiguration (bad golden case, no cases found).

Usage (from repo root):

    uv run python evals/run_evals.py --provider mock

What mock mode proves (honesty statement):

- Each golden case ships a hand-authored ``mock_response.json`` whose draft
  and critic JSON were written to be FAITHFUL to that case's checked-in
  source text. With ``--provider mock`` the case runs the real Pipeline
  against ``MockProvider(script=[draft, critic, judge])``, so the run is
  fully deterministic and offline (no network, no keys).
- Mock mode therefore validates the HARNESS + RUBRIC mechanics (case
  loading, real pipeline wiring, normalized fact matching, thresholds,
  metering, exit codes) -- it does NOT measure real extraction quality.
- Live providers (``--provider gemini|anthropic|openai|ollama``) run the
  same golden set and rubric against real LLM output; that is the run that
  measures actual extraction quality.
"""

import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from distill.llm import get_provider
from distill.llm.base import LLMPort
from distill.llm.mock_provider import MockProvider
from distill.models import CriticResult, IngestTrace, KnowledgeDraft, RawDocument
from distill.pipeline.orchestrator import Pipeline, PipelineError

# evals/ is not an installed package; make sibling module `rubric` importable
# both when run as a script and when imported by tests from the repo root.
EVALS_DIR = Path(__file__).resolve().parent
if str(EVALS_DIR) not in sys.path:
    sys.path.insert(0, str(EVALS_DIR))

import rubric  # noqa: E402

DEFAULT_GOLDEN_DIR = EVALS_DIR / "golden"
SOURCE_FILENAMES = ("source.txt", "source.html", "source.md")


class ConfigError(Exception):
    """A golden case (or the run configuration) is broken; exit code 2.

    Distinct from an eval FAILURE (exit 1): a failure means the pipeline
    output did not meet the rubric; a ConfigError means the harness cannot
    honestly run at all.
    """


@dataclass
class GoldenCase:
    """One loaded golden case: source text + expectations + mock script."""

    name: str
    source_text: str
    expected: rubric.ExpectedCase
    mock_draft: str | None = None
    mock_critic: str | None = None
    mock_judge: str | None = None


def _read_source(case_dir: Path) -> str:
    for filename in SOURCE_FILENAMES:
        path = case_dir / filename
        if path.is_file():
            return path.read_text(encoding="utf-8")
    raise ConfigError(
        f"golden case {case_dir.name!r} has no source file "
        f"(expected one of: {', '.join(SOURCE_FILENAMES)})"
    )


def _load_mock_script(case_dir: Path) -> tuple[str | None, str | None, str | None]:
    """Load and STRICTLY validate mock_response.json (if present).

    The draft/critic/judge payloads are validated against the real models at
    load time so a broken hand-authored script fails fast as a ConfigError
    instead of silently derailing the scripted call order mid-run.
    """
    path = case_dir / "mock_response.json"
    if not path.is_file():
        return None, None, None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{path} is not valid JSON: {exc}") from exc
    try:
        draft = KnowledgeDraft.model_validate(payload["draft"])
        critic = CriticResult.model_validate(payload["critic"])
        judge = (
            CriticResult.model_validate(payload["judge"]) if "judge" in payload else None
        )
    except KeyError as exc:
        raise ConfigError(f"{path} is missing required key {exc}") from exc
    except ValidationError as exc:
        raise ConfigError(f"{path} does not validate against the models: {exc}") from exc
    return (
        draft.model_dump_json(),
        critic.model_dump_json(),
        judge.model_dump_json() if judge is not None else None,
    )


def load_cases(golden_dir: Path) -> list[GoldenCase]:
    """Load every golden case (a subdirectory holding expected.json)."""
    if not golden_dir.is_dir():
        raise ConfigError(f"golden directory not found: {golden_dir}")
    cases: list[GoldenCase] = []
    for case_dir in sorted(p for p in golden_dir.iterdir() if p.is_dir()):
        expected_path = case_dir / "expected.json"
        if not expected_path.is_file():
            continue
        try:
            expected = rubric.ExpectedCase.model_validate_json(
                expected_path.read_text(encoding="utf-8")
            )
        except ValidationError as exc:
            raise ConfigError(f"{expected_path} does not validate: {exc}") from exc
        mock_draft, mock_critic, mock_judge = _load_mock_script(case_dir)
        cases.append(
            GoldenCase(
                name=case_dir.name,
                source_text=_read_source(case_dir),
                expected=expected,
                mock_draft=mock_draft,
                mock_critic=mock_critic,
                mock_judge=mock_judge,
            )
        )
    if not cases:
        raise ConfigError(f"no golden cases found under {golden_dir}")
    return cases


def build_case_provider(case: GoldenCase, provider_name: str, shared: LLMPort | None) -> LLMPort:
    """Return the LLM provider for one case.

    Mock: a fresh per-case MockProvider scripted with exactly the three
    responses the run consumes in order -- extract draft, pipeline critic,
    rubric judge (falling back to the critic verdict when no separate judge
    is authored). Live: the single shared provider.
    """
    if provider_name != "mock":
        assert shared is not None
        return shared
    if case.mock_draft is None or case.mock_critic is None:
        raise ConfigError(
            f"golden case {case.name!r} has no mock_response.json; "
            "every case needs one to run with --provider mock"
        )
    return MockProvider(
        script=[case.mock_draft, case.mock_critic, case.mock_judge or case.mock_critic]
    )


def _check_mock_script_order(case: GoldenCase, critic_threshold: float) -> None:
    """Fail fast if a scripted critic verdict would trigger a pipeline retry.

    A retry would consume script entries the case does not have, producing a
    confusing downstream parse error; a load-time ConfigError names the real
    problem instead.
    """
    assert case.mock_critic is not None
    critic = CriticResult.model_validate_json(case.mock_critic)
    if critic.confidence < critic_threshold:
        raise ConfigError(
            f"golden case {case.name!r}: scripted critic confidence "
            f"{critic.confidence} is below --critic-threshold {critic_threshold}; "
            "this would trigger a pipeline retry and exhaust the mock script. "
            "Raise the scripted confidence or lower the threshold."
        )


def run_case(
    case: GoldenCase,
    llm: LLMPort,
    *,
    critic_threshold: float,
    fact_floor: float,
    judge_floor: float,
) -> tuple[rubric.CaseResult, IngestTrace]:
    """Run one case through the REAL Pipeline, then apply the rubric.

    A PipelineError (from the pipeline or the judge) yields a FAILED
    CaseResult -- never a crashed harness -- and keeps the partial trace so
    failed runs are still metered.
    """
    raw = RawDocument(
        source_type="url",
        source_ref=f"golden://{case.name}",
        title=case.name.replace("_", " "),
        text=case.source_text,
        fetched_at=datetime.now(UTC),
    )
    pipeline = Pipeline(llm, critic_threshold=critic_threshold)
    try:
        doc, trace = pipeline.run(raw)
    except PipelineError as err:
        return rubric.failed_case(case.name, str(err)), err.partial_trace or IngestTrace()
    try:
        judge = rubric.llm_judge_faithfulness(llm, case.source_text, doc)
    except PipelineError as err:
        return rubric.failed_case(case.name, f"judge failed: {err}"), trace
    result = rubric.apply_rubric(
        case.name, doc, case.expected, judge, fact_floor=fact_floor, judge_floor=judge_floor
    )
    return result, trace


def _flag(ok: bool) -> str:
    return "ok" if ok else "FAIL"


def print_report(
    results: list[rubric.CaseResult],
    traces: list[IngestTrace],
    *,
    threshold: float,
) -> float:
    """Print the per-case table + summary; return the pass rate."""
    name_width = max(len(r.case) for r in results)
    name_width = max(name_width, len("case"))
    header = (
        f"{'case':<{name_width}}  {'schema':<6}  {'facts':<5}  "
        f"{'topics':<6}  {'points':<6}  {'judge':<5}  {'pass':<4}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        print(
            f"{result.case:<{name_width}}  {_flag(result.schema_ok):<6}  "
            f"{result.fact_score:<5.2f}  {_flag(result.topic_ok):<6}  "
            f"{_flag(result.points_ok):<6}  {result.judge_confidence:<5.2f}  "
            f"{'PASS' if result.passed else 'FAIL'}"
        )
        if result.missing_facts:
            for fact in result.missing_facts:
                print(f"{'':<{name_width}}  missing fact: {fact!r}")
        if result.error:
            print(f"{'':<{name_width}}  error: {result.error}")
    print("-" * len(header))

    passes = sum(1 for r in results if r.passed)
    pass_rate = passes / len(results)
    confidences = [r.judge_confidence for r in results]
    tokens_in = sum(t.total_tokens_in for t in traces)
    tokens_out = sum(t.total_tokens_out for t in traces)
    cost = sum(t.total_cost_usd for t in traces)

    print(f"pass rate: {passes}/{len(results)} = {pass_rate:.1%} (floor: {threshold:.1%})")
    print(f"mean judge confidence: {statistics.mean(confidences):.4f}")
    print(f"judge confidence variance (pvariance): {statistics.pvariance(confidences):.6f}")
    print(f"total tokens: in={tokens_in} out={tokens_out} total={tokens_in + tokens_out}")
    print(f"total cost: ${cost:.6f}")
    return pass_rate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the distill golden-set evals through the real pipeline."
    )
    parser.add_argument(
        "--provider",
        default="mock",
        choices=("mock", "gemini", "anthropic", "openai", "ollama"),
        help="LLM provider (default: mock -- deterministic, offline, no keys)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.99,
        help="pass-rate floor; exit 1 below it (default: 0.99)",
    )
    parser.add_argument(
        "--critic-threshold",
        type=float,
        default=0.7,
        help="pipeline critic confidence below which a bounded retry fires (default: 0.7)",
    )
    parser.add_argument(
        "--fact-threshold",
        type=float,
        default=rubric.FACT_SCORE_FLOOR,
        help=f"per-case key-fact score floor (default: {rubric.FACT_SCORE_FLOOR})",
    )
    parser.add_argument(
        "--judge-threshold",
        type=float,
        default=rubric.JUDGE_CONFIDENCE_FLOOR,
        help=(
            "per-case LLM-judge confidence floor "
            f"(default: {rubric.JUDGE_CONFIDENCE_FLOOR})"
        ),
    )
    parser.add_argument(
        "--golden-dir",
        type=Path,
        default=DEFAULT_GOLDEN_DIR,
        help="directory of golden cases (default: evals/golden)",
    )
    args = parser.parse_args(argv)

    try:
        cases = load_cases(args.golden_dir)
        shared: LLMPort | None = None
        if args.provider != "mock":
            shared = get_provider(args.provider)
        providers: list[LLMPort] = []
        for case in cases:
            provider = build_case_provider(case, args.provider, shared)
            if args.provider == "mock":
                _check_mock_script_order(case, args.critic_threshold)
            providers.append(provider)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    results: list[rubric.CaseResult] = []
    traces: list[IngestTrace] = []
    for case, provider in zip(cases, providers, strict=True):
        result, trace = run_case(
            case,
            provider,
            critic_threshold=args.critic_threshold,
            fact_floor=args.fact_threshold,
            judge_floor=args.judge_threshold,
        )
        results.append(result)
        traces.append(trace)

    print(f"provider: {args.provider}  cases: {len(cases)}")
    pass_rate = print_report(results, traces, threshold=args.threshold)
    return 0 if pass_rate >= args.threshold else 1


if __name__ == "__main__":
    sys.exit(main())
