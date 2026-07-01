from __future__ import annotations

from collections.abc import Iterator

from sda.core import ScenarioBatch


class DataModule:
    """Lightning-style container for SimPy scenario setup and batching.

    Subclass ``DataModule`` when scenario construction has state, setup, or
    reusable configuration. Implement :meth:`batches` to yield
    :class:`ScenarioBatch` objects containing ``ScenarioSpec`` items for the
    requested stage.
    """

    def prepare_data(self) -> None:
        """Prepare shared data before setup.

        Override this hook for one-time data work such as downloading,
        extracting, or loading shared history.
        """

    def setup(self, stage: str | None = None) -> None:
        """Set up stage-specific scenario state before evaluation."""

    def batches(self, stage: str = "evaluate") -> Iterator[ScenarioBatch]:
        """Yield scenario batches for ``stage``."""
        raise NotImplementedError


__all__ = [
    "DataModule",
]
