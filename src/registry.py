from __future__ import annotations
from collections import defaultdict
from typing import Callable, Dict

_REGISTRY: Dict[str, Dict[str, Callable]] = defaultdict(dict)

def register(kind: str, name: str):
    """Decorator to register builders under a kind/name."""
    def deco(fn: Callable):
        if name in _REGISTRY[kind]:
            raise KeyError(f"Duplicate registration: {kind}:{name}")
        _REGISTRY[kind][name] = fn
        return fn
    return deco

def build(kind: str, name: str, **kwargs):
    try:
        return _REGISTRY[kind][name](**kwargs)
    except KeyError as e:
        raise KeyError(f"No builder for {kind}:{name}") from e