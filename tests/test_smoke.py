"""Phase 0 smoke test: the package imports and reports its version."""

import distill


def test_import_and_version() -> None:
    assert distill.__version__ == "0.1.0"
