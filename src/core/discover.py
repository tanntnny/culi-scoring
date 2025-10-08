"""
Automatic discovery/importer for project submodules.
"""

from __future__ import annotations

import pkgutil
import importlib
from typing import Iterable, Optional, List


def _import_module_safe(module_name: str) -> None:
    try:
        importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - best-effort runtime behavior
        print(f"[discover] warning: failed to import {module_name!r}: {exc}")


def discover_packages(package_root: str = "src", package_names: Optional[Iterable[str]] = None) -> List[str]:
    touched: List[str] = []

    try:
        root = importlib.import_module(package_root)
    except Exception as exc:  # pragma: no cover
        raise ImportError(f"Cannot import package root {package_root!r}: {exc}")

    if package_names is None:
        try:
            package_names = [name for _, name, ispkg in pkgutil.iter_modules(root.__path__) if ispkg]
        except Exception:
            package_names = []

    for name in package_names:
        full_pkg = f"{package_root}.{name}"
        try:
            mod = importlib.import_module(full_pkg)
            touched.append(full_pkg)
        except Exception as exc:  # pragma: no cover - keep scanning other packages
            print(f"[discover] warning: failed to import package {full_pkg!r}: {exc}")
            continue

        pkg_path = getattr(mod, "__path__", None)
        if pkg_path is None:
            continue

        for finder, modname, ispkg in pkgutil.walk_packages(pkg_path, prefix=mod.__name__ + "."):
            _import_module_safe(modname)
            touched.append(modname)

    return touched


def discover_default() -> List[str]:
    default_pkgs = ("data", "models", "optim", "pipeline", "downloads", "tasks", "metrics")
    return discover_packages("src", default_pkgs)


if __name__ == "__main__":  # quick CLI for debugging
    print("Running discovery (defaults)...")
    imported = discover_default()
    print(f"Discovered/imported {len(imported)} modules (sample): {imported[:10]}")