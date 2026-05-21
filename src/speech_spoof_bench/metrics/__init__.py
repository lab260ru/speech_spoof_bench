"""Pluggable metric registry.

Adding a new metric: drop a file under this package that calls
``@register_metric(...)`` at import time. The dataset's ``eval.yaml``
references metrics by their ``id``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

ScoresMap = dict[str, float]
LabelsMap = dict[str, int]


@dataclass(frozen=True)
class MetricResult:
    value: float
    extras: dict[str, Any] = field(default_factory=dict)


MetricFn = Callable[[ScoresMap, LabelsMap], MetricResult]


@dataclass(frozen=True)
class MetricSpec:
    id: str
    display_name: str
    lower_is_better: bool
    requires_audio: bool
    fn: MetricFn


_REGISTRY: dict[str, MetricSpec] = {}


def register_metric(
    *,
    id: str,
    display_name: str,
    lower_is_better: bool,
    requires_audio: bool = False,
) -> Callable[[MetricFn], MetricFn]:
    def decorator(fn: MetricFn) -> MetricFn:
        if id in _REGISTRY:
            raise ValueError(f"metric id {id!r} already registered")
        _REGISTRY[id] = MetricSpec(
            id=id,
            display_name=display_name,
            lower_is_better=lower_is_better,
            requires_audio=requires_audio,
            fn=fn,
        )
        return fn

    return decorator


def get_metric(id: str) -> MetricSpec:
    try:
        return _REGISTRY[id]
    except KeyError:
        raise KeyError(f"metric id {id!r} is not registered") from None


def list_metrics() -> list[MetricSpec]:
    return list(_REGISTRY.values())


def is_registered(id: str) -> bool:
    return id in _REGISTRY
