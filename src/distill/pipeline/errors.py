"""PipelineError -- honest failure reporting with stage name + cause."""


class PipelineError(Exception):
    """Raised when a pipeline stage cannot complete.

    Carries the failing stage name and the underlying cause so callers
    (API layer, eval harness) can report failures honestly instead of
    swallowing them.
    """

    def __init__(self, stage: str, cause: Exception | None = None) -> None:
        self.stage = stage
        self.cause = cause
        detail = f": {cause}" if cause is not None else ""
        super().__init__(f"pipeline stage '{stage}' failed{detail}")
