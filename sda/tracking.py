from __future__ import annotations

from collections.abc import Mapping, Sequence
import importlib
import math
import re
from typing import Any


_DEFAULT_SUMMARY_STATS = (
    "count",
    "mean",
    "std",
    "min",
    "p50",
    "p90",
    "p95",
    "p99",
    "max",
)
_INVALID_MLFLOW_KEY_CHARS = re.compile(r"[^A-Za-z0-9_.\-/ ]+")
_MAX_MLFLOW_KEY_LENGTH = 250


class MLflowTracker:
    """Log simulation result summaries to an MLflow tracking run.

    MLflow is imported lazily, so normal ``sda`` usage does not require the
    optional dependency. Install it with ``sda-simulator-v2[mlflow]`` before
    using this tracker.
    """

    def __init__(
        self,
        *,
        experiment_name: str | None = None,
        run_name: str | None = None,
        tracking_uri: str | None = None,
        params: Mapping[str, Any] | None = None,
        tags: Mapping[str, Any] | None = None,
        metric_prefix: str = "",
        summary_stats: Sequence[str] | None = None,
        nested: bool = False,
    ) -> None:
        """Configure MLflow logging for one simulation evaluation.

        Parameters
        ----------
        experiment_name
            Optional MLflow experiment name. Missing experiments are created by
            MLflow's ``set_experiment`` call.
        run_name
            Optional run name for the MLflow run.
        tracking_uri
            Optional MLflow tracking URI, for example a local ``file:`` URI or
            a tracking server URL.
        params
            Run parameters to log alongside simulator context.
        tags
            Run tags to attach to the MLflow run.
        metric_prefix
            Optional prefix prepended to logged summary metric keys.
        summary_stats
            Summary statistic names to log from each metric distribution. The
            default logs ``count``, ``mean``, ``std``, common percentiles, and
            min/max.
        nested
            Passed through to ``mlflow.start_run`` for nested run support.
        """
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.tracking_uri = tracking_uri
        self.params = dict(params or {})
        self.tags = dict(tags or {})
        self.metric_prefix = metric_prefix
        self.summary_stats = (
            _DEFAULT_SUMMARY_STATS
            if summary_stats is None
            else tuple(summary_stats)
        )
        self.nested = nested

    def log_result(
        self,
        result: Any,
        *,
        params: Mapping[str, Any] | None = None,
        tags: Mapping[str, Any] | None = None,
    ) -> str | None:
        """Log a ``SimulationResult`` summary and return the MLflow run id."""
        mlflow = _import_mlflow()
        if self.tracking_uri is not None:
            mlflow.set_tracking_uri(self.tracking_uri)
        if self.experiment_name is not None:
            mlflow.set_experiment(self.experiment_name)

        run_params = _string_mapping({**self.params, **dict(params or {})})
        run_tags = _string_mapping({**self.tags, **dict(tags or {})})
        metrics = self.summary_metrics(result.summary())

        with mlflow.start_run(run_name=self.run_name, nested=self.nested) as run:
            if run_params:
                mlflow.log_params(run_params)
            if run_tags:
                mlflow.set_tags(run_tags)
            if metrics:
                mlflow.log_metrics(metrics)
            return getattr(getattr(run, "info", None), "run_id", None)

    def summary_metrics(
        self,
        summary: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, float]:
        """Return finite MLflow metric values from a result summary."""
        metrics: dict[str, float] = {}
        for metric_name, stats in summary.items():
            for stat in self.summary_stats:
                if stat not in stats:
                    continue
                value = _finite_float(stats[stat])
                if value is None:
                    continue
                key = _mlflow_key(self.metric_prefix, metric_name, stat)
                if key in metrics:
                    raise ValueError(
                        f"duplicate MLflow metric key after sanitizing: {key!r}"
                    )
                metrics[key] = value
        return metrics


def _import_mlflow():
    try:
        return importlib.import_module("mlflow")
    except ModuleNotFoundError as error:
        if error.name != "mlflow":
            raise
        raise ImportError(
            "MLflow tracking requires the optional 'mlflow' dependency. "
            "Install it with 'pip install sda-simulator-v2[mlflow]' or "
            "'pip install mlflow'."
        ) from error


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _string_mapping(values: Mapping[str, Any]) -> dict[str, str]:
    return {
        _mlflow_key(key): _stringify(value)
        for key, value in values.items()
    }


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    return str(value)


def _mlflow_key(*parts: object) -> str:
    key = ".".join(str(part).strip(".") for part in parts if str(part))
    sanitized = _INVALID_MLFLOW_KEY_CHARS.sub("_", key).strip()
    if not sanitized:
        raise ValueError("MLflow keys must not be empty")
    if len(sanitized) > _MAX_MLFLOW_KEY_LENGTH:
        raise ValueError(
            f"MLflow key {sanitized!r} exceeds {_MAX_MLFLOW_KEY_LENGTH} characters"
        )
    return sanitized


__all__ = [
    "MLflowTracker",
]
