"""Phase 2 tests: LLM provider layer (MockProvider only -- no network)."""

import json

import pytest

from distill.config import Config
from distill.llm import get_provider
from distill.llm.base import COST_TABLE, LLMPort, LLMResponse, estimate_cost
from distill.llm.mock_provider import MockProvider
from distill.models import CriticResult, KnowledgeDraft


class TestMockProviderDeterminism:
    def test_same_prompt_twice_identical_response(self) -> None:
        provider = MockProvider()
        first = provider.complete("summarize this article about databases")
        second = provider.complete("summarize this article about databases")
        assert first == second

    def test_determinism_across_instances(self) -> None:
        a = MockProvider().complete("hello world")
        b = MockProvider().complete("hello world")
        assert a == b


class TestMockProviderCannedResponses:
    def test_extract_marker_returns_valid_knowledge_draft_json(self) -> None:
        provider = MockProvider()
        response = provider.complete("TASK: EXTRACT\n\nSome source text here.")
        draft = KnowledgeDraft.model_validate_json(response.text)
        assert draft.summary
        assert draft.key_points
        assert draft.entities
        assert draft.topics

    def test_critic_marker_returns_valid_critic_result_json(self) -> None:
        provider = MockProvider()
        response = provider.complete("TASK: CRITIC\n\nDraft vs source.")
        result = CriticResult.model_validate_json(response.text)
        assert result.confidence >= 0.7
        assert result.faithful is True

    def test_unmarked_prompt_returns_fixed_echo(self) -> None:
        provider = MockProvider()
        response = provider.complete("just chatting")
        # Not JSON -- the fixed fallback string.
        with pytest.raises(json.JSONDecodeError):
            json.loads(response.text)
        assert response.text == MockProvider.DEFAULT_RESPONSE

    def test_responses_override_takes_priority(self) -> None:
        provider = MockProvider(responses={"MAGIC": "overridden"})
        response = provider.complete("prompt containing MAGIC token")
        assert response.text == "overridden"

    def test_earliest_marker_wins_over_embedded_marker(self) -> None:
        # A critic prompt whose embedded SOURCE text happens to contain the
        # extract marker must still resolve as a critic prompt, because the
        # template's own "TASK: CRITIC" appears earlier in the prompt.
        prompt = (
            "TASK: CRITIC\n\nJudge the draft.\n\nSOURCE:\n"
            "An article explaining what TASK: EXTRACT means in this pipeline.\n\n"
            "DRAFT (JSON):\n{}"
        )
        response = MockProvider().complete(prompt)
        result = CriticResult.model_validate_json(response.text)
        assert result.faithful is True

    def test_repair_marker_returns_default_not_canned_draft(self) -> None:
        # A repair prompt embedding invalid output that contains an extract
        # marker must NOT be rescued by the canned draft.
        prompt = "TASK: REPAIR\n\nThe invalid response was:\n\nTASK: EXTRACT {bad"
        response = MockProvider().complete(prompt)
        assert response.text == MockProvider.DEFAULT_RESPONSE


class TestMockProviderScript:
    def test_script_queue_returns_responses_in_order(self) -> None:
        provider = MockProvider(script=["one", "two", "three"])
        assert provider.complete("a").text == "one"
        assert provider.complete("b").text == "two"
        assert provider.complete("c").text == "three"

    def test_script_exhausted_falls_back_to_keyed_behavior(self) -> None:
        provider = MockProvider(script=["only"])
        assert provider.complete("x").text == "only"
        # After the queue drains, normal keyed behavior resumes.
        assert provider.complete("x").text == MockProvider.DEFAULT_RESPONSE

    def test_script_supports_malformed_then_valid_sequence(self) -> None:
        malformed = "{not valid json"
        provider = MockProvider(script=[malformed, "TASK-agnostic ok"])
        first = provider.complete("TASK: EXTRACT")
        assert first.text == malformed


class TestLLMResponseFields:
    def test_fields_populated_for_nonempty_prompt(self) -> None:
        response = MockProvider().complete("a reasonably sized prompt for counting")
        assert isinstance(response, LLMResponse)
        assert response.tokens_in > 0
        assert response.tokens_out > 0
        assert response.cost_usd == 0.0

    def test_tokens_in_scales_with_prompt_length(self) -> None:
        provider = MockProvider()
        short = provider.complete("hi there padding")
        long = provider.complete("hi there padding " * 50)
        assert long.tokens_in > short.tokens_in

    def test_system_prompt_counts_toward_tokens_in(self) -> None:
        provider = MockProvider()
        bare = provider.complete("same prompt")
        with_system = provider.complete("same prompt", system="You are a careful extractor." * 5)
        assert with_system.tokens_in > bare.tokens_in

    def test_temperature_is_accepted_and_ignored(self) -> None:
        provider = MockProvider()
        assert provider.complete("same prompt", temperature=0.0) == provider.complete(
            "same prompt"
        )


class TestCostTable:
    @pytest.mark.parametrize(
        "model",
        [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "claude-sonnet-4-5",
            "claude-haiku-4-5",
            "gpt-4o-mini",
            "gpt-4o",
        ],
    )
    def test_known_model_costs_positive(self, model: str) -> None:
        assert model in COST_TABLE
        assert estimate_cost(model, tokens_in=1_000_000, tokens_out=1_000_000) > 0.0

    def test_mock_and_ollama_are_zero_cost(self) -> None:
        assert estimate_cost("mock", 1_000_000, 1_000_000) == 0.0
        assert estimate_cost("ollama", 1_000_000, 1_000_000) == 0.0

    def test_unknown_model_returns_zero_not_crash(self) -> None:
        assert estimate_cost("model-that-does-not-exist", 500, 500) == 0.0

    def test_cost_math_matches_table(self) -> None:
        usd_in, usd_out = COST_TABLE["gpt-4o"]
        expected = usd_in * 2.0 + usd_out * 0.5
        assert estimate_cost("gpt-4o", 2_000_000, 500_000) == pytest.approx(expected)


class TestFactory:
    def test_default_is_mock_when_env_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DISTILL_PROVIDER", raising=False)
        assert isinstance(get_provider(), MockProvider)

    def test_explicit_mock_name(self) -> None:
        assert isinstance(get_provider("mock"), MockProvider)

    def test_unknown_name_raises_value_error_listing_valid_names(self) -> None:
        with pytest.raises(ValueError, match="mock"):
            get_provider("nope")

    def test_env_var_selects_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DISTILL_PROVIDER", "mock")
        assert isinstance(get_provider(), MockProvider)


class TestProtocolConformance:
    def test_mock_provider_satisfies_llm_port(self) -> None:
        assert isinstance(MockProvider(), LLMPort)


class TestConfig:
    def test_from_env_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for var in (
            "DISTILL_PROVIDER",
            "DISTILL_MODEL",
            "DISTILL_CRITIC_THRESHOLD",
            "OLLAMA_BASE_URL",
        ):
            monkeypatch.delenv(var, raising=False)
        cfg = Config.from_env()
        assert cfg.provider == "mock"
        assert cfg.model is None
        assert cfg.critic_threshold == 0.7
        assert cfg.ollama_base_url == "http://localhost:11434"

    def test_from_env_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DISTILL_PROVIDER", "gemini")
        monkeypatch.setenv("DISTILL_MODEL", "gemini-2.5-pro")
        monkeypatch.setenv("DISTILL_CRITIC_THRESHOLD", "0.85")
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://10.0.0.5:11434")
        cfg = Config.from_env()
        assert cfg.provider == "gemini"
        assert cfg.model == "gemini-2.5-pro"
        assert cfg.critic_threshold == 0.85
        assert cfg.ollama_base_url == "http://10.0.0.5:11434"


class TestLazySdkImports:
    """Importing distill.llm must work without any provider SDK installed."""

    def test_importing_llm_package_needs_no_sdk(self) -> None:
        # If module-level SDK imports existed, collection itself would fail
        # in this SDK-free environment. Importing the concrete provider
        # modules must also succeed.
        import distill.llm.anthropic_provider  # noqa: F401
        import distill.llm.gemini_provider  # noqa: F401
        import distill.llm.ollama_provider  # noqa: F401
        import distill.llm.openai_provider  # noqa: F401
