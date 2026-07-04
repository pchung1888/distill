"""Daily rate limiting for the public demo (distill-demo frontend).

This is deliberately NOT part of the core pipeline's "no database in v1"
decision (HANDOFF Section 3) -- it exists only to bound real-money spend when
distill is reachable from a public browser. Storage is a tiny SQLite table
holding daily counters, not application data.

Two caps, both reset at UTC midnight:
- Per-visitor: N single-ingest runs and M compare runs per day.
- Global: a total USD spend ceiling shared across all visitors.

Identity is a random id stored in an httponly cookie (no login). This is
easy to clear (incognito, cookie reset), which is an accepted tradeoff for a
portfolio demo, not a production abuse-prevention system.
"""

import sqlite3
import threading
from dataclasses import dataclass
from datetime import UTC, datetime

VISITOR_COOKIE = "distill_visitor"


class RateLimitExceeded(Exception):
    """Raised when a visitor or global cap would be exceeded.

    `kind` is one of "visitor_single", "visitor_compare", "global_budget" --
    the API maps this to a 429 response the frontend can key off of to show
    the curated-examples fallback (never a silent switch to MockProvider).
    """

    def __init__(self, kind: str, message: str) -> None:
        self.kind = kind
        super().__init__(message)


@dataclass
class RateLimitConfig:
    daily_single_runs: int = 5
    daily_compares: int = 1
    global_daily_budget_usd: float = 2.0


class RateLimiter:
    """SQLite-backed daily counters. One instance per app; owns one connection."""

    def __init__(self, db_path: str = ":memory:", config: RateLimitConfig | None = None) -> None:
        self.config = config or RateLimitConfig()
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS visitor_usage (
                visitor_id TEXT NOT NULL,
                day TEXT NOT NULL,
                single_runs INTEGER NOT NULL DEFAULT 0,
                compares INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (visitor_id, day)
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS global_spend (
                day TEXT PRIMARY KEY,
                spent_usd REAL NOT NULL DEFAULT 0.0
            )
            """
        )
        self._conn.commit()

    @staticmethod
    def _today() -> str:
        return datetime.now(UTC).date().isoformat()

    def _spent_today(self, today: str) -> float:
        row = self._conn.execute(
            "SELECT spent_usd FROM global_spend WHERE day = ?", (today,)
        ).fetchone()
        return row[0] if row else 0.0

    def check_and_reserve(self, visitor_id: str, kind: str) -> None:
        """Raise RateLimitExceeded if the visitor/global cap is already hit,
        otherwise record this attempt immediately (counts the attempt even if
        the pipeline call that follows later fails -- it still cost tokens up
        to the point of failure, and this prevents retry-looping the cap)."""
        if kind not in ("single", "compare"):
            raise ValueError(f"unknown rate-limit kind: {kind!r}")
        today = self._today()
        with self._lock:
            row = self._conn.execute(
                "SELECT single_runs, compares FROM visitor_usage WHERE visitor_id = ? AND day = ?",
                (visitor_id, today),
            ).fetchone()
            single_runs, compares = row if row else (0, 0)

            if kind == "single" and single_runs >= self.config.daily_single_runs:
                raise RateLimitExceeded(
                    "visitor_single",
                    f"daily single-run limit ({self.config.daily_single_runs}) reached",
                )
            if kind == "compare" and compares >= self.config.daily_compares:
                raise RateLimitExceeded(
                    "visitor_compare",
                    f"daily compare limit ({self.config.daily_compares}) reached",
                )
            if self._spent_today(today) >= self.config.global_daily_budget_usd:
                raise RateLimitExceeded(
                    "global_budget", "daily global demo budget reached"
                )

            if kind == "single":
                single_runs += 1
            else:
                compares += 1
            self._conn.execute(
                """
                INSERT INTO visitor_usage (visitor_id, day, single_runs, compares)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(visitor_id, day) DO UPDATE SET
                    single_runs = excluded.single_runs,
                    compares = excluded.compares
                """,
                (visitor_id, today, single_runs, compares),
            )
            self._conn.commit()

    def record_spend(self, usd: float) -> None:
        """Add `usd` to today's global spend total. Called after a run
        completes (or partially completes) with its real IngestTrace cost."""
        today = self._today()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO global_spend (day, spent_usd) VALUES (?, ?)
                ON CONFLICT(day) DO UPDATE SET spent_usd = spent_usd + excluded.spent_usd
                """,
                (today, usd),
            )
            self._conn.commit()

    def remaining(self, visitor_id: str) -> dict[str, float | int]:
        """Snapshot of what `visitor_id` has left today (for the frontend to
        show 'N runs left today' without needing to guess from 429s)."""
        today = self._today()
        row = self._conn.execute(
            "SELECT single_runs, compares FROM visitor_usage WHERE visitor_id = ? AND day = ?",
            (visitor_id, today),
        ).fetchone()
        single_runs, compares = row if row else (0, 0)
        spent = self._spent_today(today)
        return {
            "single_runs_left": max(0, self.config.daily_single_runs - single_runs),
            "compares_left": max(0, self.config.daily_compares - compares),
            "global_budget_left_usd": max(0.0, self.config.global_daily_budget_usd - spent),
        }
