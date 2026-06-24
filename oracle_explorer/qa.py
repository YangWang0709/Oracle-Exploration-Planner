"""QA checks for oracle maps, paths, trajectories, and debug artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .grid import GridIndex, find_path_violations


@dataclass
class QAReport:
    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "errors": self.errors,
            "metrics": self.metrics,
            "passed": self.passed,
            "warnings": self.warnings,
        }


def _nonempty_grid(name: str, grid: np.ndarray | None, errors: list[str]) -> int:
    if grid is None:
        errors.append(f"{name} is missing")
        return 0
    arr = np.asarray(grid)
    if arr.size == 0:
        errors.append(f"{name} is empty")
        return 0
    count = int(np.asarray(arr, dtype=bool).sum())
    if count == 0:
        errors.append(f"{name} has no true cells")
    return count


def qa_map_path(
    *,
    occupancy_grid: np.ndarray | None,
    traversable_grid: np.ndarray | None,
    reachable_grid: np.ndarray | None,
    path: Iterable[GridIndex],
    trajectory: Sequence[dict[str, object]] | None = None,
    final_coverage: float | None = None,
    coverage_threshold: float | None = None,
    debug_pngs: Sequence[str | Path] | None = None,
    fallback_used: bool | None = None,
    fallback_allowed: bool = True,
    min_reachable_cells: int | None = None,
    min_reachable_ratio: float | None = None,
    occupancy_ratio_bounds: tuple[float, float] | None = None,
    traversable_ratio_bounds: tuple[float, float] | None = None,
    object_summary_path: str | Path | None = None,
) -> QAReport:
    errors: list[str] = []
    warnings: list[str] = []
    metrics: dict[str, object] = {}

    occupancy_count = _nonempty_grid("occupancy_grid", occupancy_grid, errors)
    traversable_count = _nonempty_grid("traversable_grid", traversable_grid, errors)
    reachable_count = _nonempty_grid("reachable_mask", reachable_grid, errors)
    metrics.update(
        {
            "occupancy_true_cells": occupancy_count,
            "reachable_true_cells": reachable_count,
            "traversable_true_cells": traversable_count,
        }
    )

    grid_size = 0
    if occupancy_grid is not None:
        grid_size = int(np.asarray(occupancy_grid).size)
        metrics["grid_cells"] = grid_size
    if grid_size > 0:
        occupancy_ratio = occupancy_count / grid_size
        traversable_ratio = traversable_count / grid_size
        reachable_ratio = reachable_count / grid_size
        metrics.update(
            {
                "occupancy_ratio": occupancy_ratio,
                "reachable_ratio": reachable_ratio,
                "traversable_ratio": traversable_ratio,
            }
        )
        if occupancy_ratio_bounds is not None:
            lo, hi = occupancy_ratio_bounds
            if not (lo <= occupancy_ratio <= hi):
                errors.append(
                    f"occupancy ratio {occupancy_ratio:.6f} outside bounds [{lo:.6f}, {hi:.6f}]"
                )
        if traversable_ratio_bounds is not None:
            lo, hi = traversable_ratio_bounds
            if not (lo <= traversable_ratio <= hi):
                errors.append(
                    f"traversable ratio {traversable_ratio:.6f} outside bounds [{lo:.6f}, {hi:.6f}]"
                )
        if min_reachable_ratio is not None and reachable_ratio < min_reachable_ratio:
            errors.append(
                f"reachable ratio {reachable_ratio:.6f} below minimum {min_reachable_ratio:.6f}"
            )

    if min_reachable_cells is not None and reachable_count < min_reachable_cells:
        errors.append(f"reachable cells {reachable_count} below minimum {min_reachable_cells}")

    if fallback_used is not None:
        metrics["fallback_used"] = bool(fallback_used)
        if fallback_used and not fallback_allowed:
            errors.append("fallback_used=true is not allowed for this QA run")

    if object_summary_path is not None:
        summary = Path(object_summary_path)
        if not summary.exists():
            errors.append(f"object classification summary missing: {summary}")
        else:
            metrics["object_classification_summary"] = summary.as_posix()

    path_list = [(int(c[0]), int(c[1])) for c in path]
    metrics["path_cells"] = len(path_list)
    if not path_list:
        errors.append("path is empty")

    if traversable_grid is not None and path_list:
        violations = find_path_violations(path_list, np.asarray(traversable_grid, dtype=bool))
        if violations:
            errors.append(f"path has {len(violations)} traversability violations")
            metrics["path_violations"] = violations[:20]

    if trajectory is not None:
        metrics["trajectory_frames"] = len(trajectory)
        if len(trajectory) == 0:
            errors.append("trajectory is empty")

    if final_coverage is not None:
        metrics["final_coverage"] = float(final_coverage)
    if coverage_threshold is not None:
        metrics["coverage_threshold"] = float(coverage_threshold)
    if final_coverage is not None and coverage_threshold is not None:
        if final_coverage < coverage_threshold:
            errors.append(
                f"final coverage {final_coverage:.6f} is below threshold {coverage_threshold:.6f}"
            )

    for png in debug_pngs or []:
        path_obj = Path(png)
        if not path_obj.exists():
            errors.append(f"debug png missing: {path_obj}")
        else:
            metrics.setdefault("debug_pngs", []).append(path_obj.as_posix())

    return QAReport(passed=not errors, errors=errors, warnings=warnings, metrics=metrics)
