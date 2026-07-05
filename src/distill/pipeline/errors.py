"""PipelineError -- honest failure reporting with stage name + cause."""

from distill.models import IngestTrace


class PipelineError(Exception):
    """Raised when a pipeline stage cannot complete.

    Carries the failing stage name, the underlying cause, and the partial
    IngestTrace of every LLM call made before the failure, so callers
    (API layer, eval harness) can report failures -- and their token cost --
    honestly instead of swallowing them.
    """

    def __init__(
        self,
        stage: str,
        cause: Exception | None = None,
        partial_trace: IngestTrace | None = None,
    ) -> None:
        self.stage = stage
        self.cause = cause
        self.partial_trace = partial_trace
        detail = f": {cause}" if cause is not None else ""
        super().__init__(f"pipeline stage '{stage}' failed{detail}")
