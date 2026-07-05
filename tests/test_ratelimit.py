"""Unit tests for the public-demo rate limiter (SQLite-backed daily caps).

See src/distill/api/ratelimit.py module docstring for design rationale:
this is demo-safety plumbing, not the core pipeline's data layer.
"""

import pytest

from distill.api.ratelimit import RateLimitConfig, RateLimiter, RateLimitExceeded


def make_limiter(**overrides: float | int) -> RateLimiter:
    config = RateLimitConfig(
        daily_single_runs=overrides.get("daily_single_runs", 2),
        daily_compares=overrides.get("daily_compares", 1),
        global_daily_budget_usd=overrides.get("global_daily_budget_usd", 1.0),
    )
    return RateLimiter(db_path=":memory:", config=config)


class TestVisitorCaps:
    def test_single_run_allowed_up_to_daily_cap(self) -> None:
        limiter = make_limiter(daily_single_runs=2)
        limiter.check_and_reserve("visitor-a", "single")
        limiter.check_and_reserve("visitor-a", "single")  # 2nd allowed, cap is 2

    def test_single_run_blocked_after_daily_cap(self) -> None:
        limiter = make_limiter(daily_single_runs=2)
        limiter.check_and_reserve("visitor-a", "single")
        limiter.check_and_reserve("visitor-a", "single")
        with pytest.raises(RateLimitExceeded) as excinfo:
            limiter.check_and_reserve("visitor-a", "single")
        assert excinfo.value.kind == "visitor_single"

    def test_compare_blocked_after_daily_cap(self) -> None:
        limiter = make_limiter(daily_compares=1)
        limiter.check_and_reserve("visitor-a", "compare")
        with pytest.raises(RateLimitExceeded) as excinfo:
            limiter.check_and_reserve("visitor-a", "compare")
        assert excinfo.value.kind == "visitor_compare"

    def test_single_and_compare_caps_are_independent(self) -> None:
        limiter = make_limiter(daily_single_runs=1, daily_compares=1)
        limiter.check_and_reserve("visitor-a", "single")
        limiter.check_and_reserve("visitor-a", "compare")  # separate bucket, not blocked

    def test_caps_are_per_visitor(self) -> None:
        limiter = make_limiter(daily_single_runs=1)
        limiter.check_and_reserve("visitor-a", "single")
        limiter.check_and_reserve("visitor-b", "single")  # different visitor, not blocked


class TestGlobalBudget:
    def test_blocked_once_global_budget_spent(self) -> None:
        limiter = make_limiter(daily_single_runs=10, global_daily_budget_usd=0.05)
        limiter.check_and_reserve("visitor-a", "single")
        limiter.record_spend(0.05)
        with pytest.raises(RateLimitExceeded) as excinfo:
            limiter.check_and_reserve("visitor-a", "single")
        assert excinfo.value.kind == "global_budget"

    def test_global_budget_shared_across_visitors(self) -> None:
        limiter = make_limiter(daily_single_runs=10, global_daily_budget_usd=0.05)
        limiter.check_and_reserve("visitor-a", "single")
        limiter.record_spend(0.05)
        with pytest.raises(RateLimitExceeded) as excinfo:
            limiter.check_and_reserve("visitor-b", "single")
        assert excinfo.value.kind == "global_budget"

    def test_spend_accumulates_across_multiple_records(self) -> None:
        limiter = make_limiter(daily_single_runs=10, global_daily_budget_usd=0.05)
        limiter.record_spend(0.02)
        limiter.record_spend(0.02)
        limiter.check_and_reserve("visitor-a", "single")  # 0.04 spent, still under 0.05
        limiter.record_spend(0.02)  # now 0.06, over budget
        with pytest.raises(RateLimitExceeded):
            limiter.check_and_reserve("visitor-a", "single")


class TestRemaining:
    def test_remaining_reflects_usage(self) -> None:
        limiter = make_limiter(daily_single_runs=3, daily_compares=1, global_daily_budget_usd=1.0)
        limiter.check_and_reserve("visitor-a", "single")
        limiter.record_spend(0.10)
        snapshot = limiter.remaining("visitor-a")
        assert snapshot["single_runs_left"] == 2
        assert snapshot["compares_left"] == 1
        assert snapshot["global_budget_left_usd"] == pytest.approx(0.90)

    def test_remaining_never_goes_negative(self) -> None:
        limiter = make_limiter(daily_single_runs=1, global_daily_budget_usd=0.01)
        limiter.check_and_reserve("visitor-a", "single")
        limiter.record_spend(0.05)
        snapshot = limiter.remaining("visitor-a")
        assert snapshot["single_runs_left"] == 0
        assert snapshot["global_budget_left_usd"] == 0.0

    def test_unused_visitor_sees_full_caps(self) -> None:
        limiter = make_limiter(daily_single_runs=5, daily_compares=1, global_daily_budget_usd=2.0)
        snapshot = limiter.remaining("brand-new-visitor")
        assert snapshot == {
            "single_runs_left": 5,
            "compares_left": 1,
            "global_budget_left_usd": 2.0,
        }


def test_invalid_kind_raises_value_error() -> None:
    limiter = make_limiter()
    with pytest.raises(ValueError, match="unknown rate-limit kind"):
        limiter.check_and_reserve("visitor-a", "bogus")
