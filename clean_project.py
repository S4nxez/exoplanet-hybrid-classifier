#!/usr/bin/env python3
"""Workspace hygiene helper (safe by default)."""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

# Only remove cache artifacts when explicitly requested.
DEFAULT_TRASH_PATTERNS = (
    "**/__pycache__",
    "**/*.pyc",
    "**/.ipynb_checkpoints",
    ".pytest_cache",
    ".ruff_cache",
)

# Placeholder modules that still need real implementations.
PLACEHOLDER_FILES = (
    Path("src/models/mission_models.py"),
    Path("src/models/mission_models_fixed.py"),
    Path("src/models/partial_coverage.py"),
    Path("src/models/total_coverage.py"),
    Path("src/models/tensorflow_hybrid.py"),
    Path("src/trainers/multi_mission_trainer.py"),
    Path("src/trainers/partial_trainer.py"),
    Path("src/trainers/total_trainer.py"),
    Path("src/evaluators/model_evaluators.py"),
)


@dataclass
class PlaceholderInfo:
    path: Path
    bytes: int


def detect_placeholders(root: Path) -> List[PlaceholderInfo]:
    results: List[PlaceholderInfo] = []
    for rel_path in PLACEHOLDER_FILES:
        path = root / rel_path
        if path.exists():
            size = path.stat().st_size
            if size < 200:  # empty or TODO-only
                results.append(PlaceholderInfo(path=path, bytes=size))
    return results


def iter_trash(root: Path, patterns: Iterable[str]) -> Iterable[Path]:
    for pattern in patterns:
        yield from root.glob(pattern)


def purge(paths: Iterable[Path]):
    removed = 0
    for target in paths:
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
        else:
            target.unlink(missing_ok=True)
        removed += 1
    return removed


def main():
    parser = argparse.ArgumentParser(description="Audit or cleanup temporary artifacts safely.")
    parser.add_argument(
        "--delete-temp",
        action="store_true",
        help="Remove common cache directories (pyc, __pycache__, pytest caches).",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    print("\n== Proyecto: exoplanet-hybrid-classifier ==")

    placeholders = detect_placeholders(root)
    if placeholders:
        print("\nArchivos placeholder detectados (recomendado reemplazarlos o documentarlos):")
        for info in placeholders:
            print(f"  - {info.path} ({info.bytes} bytes)")
    else:
        print("\nNo se encontraron placeholders conocidos en src/.")

    if args.delete_temp:
        trash = list(iter_trash(root, DEFAULT_TRASH_PATTERNS))
        if trash:
            removed = purge(trash)
            print(f"\nSe eliminaron {removed} artefactos temporales.")
        else:
            print("\nNo se encontraron artefactos temporales para eliminar.")
    else:
        print("\nModo solo-auditoria: no se elimino ningun archivo.")
        print("Use --delete-temp para borrar caches seguros (no afecta codigo ni modelos).")


if __name__ == "__main__":
    main()
