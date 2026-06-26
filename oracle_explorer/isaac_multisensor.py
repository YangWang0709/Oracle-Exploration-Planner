"""Isaac LiDAR capability checks and real LaserScan collection helpers."""

from __future__ import annotations

import importlib
import asyncio
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from oracle_explorer.io_utils import ensure_dir, write_json
from oracle_explorer.sensors.lidar import laserscan_stats, pointcloud_to_laserscan, save_laserscan, save_laserscan_npy


TRUE_ISAAC_BACKENDS = ("isaac_rtx_lidar", "isaac_range_sensor_lidar", "isaac_physx_lidar")
ALL_BACKENDS = (*TRUE_ISAAC_BACKENDS, "custom_usd_raycast_laserscan")
BACKEND_ALIASES = {
    "auto": "auto",
    "rtx": "isaac_rtx_lidar",
    "isaac_rtx_lidar": "isaac_rtx_lidar",
    "range_sensor": "isaac_range_sensor_lidar",
    "isaac_range_sensor_lidar": "isaac_range_sensor_lidar",
    "physx": "isaac_physx_lidar",
    "raycast": "isaac_physx_lidar",
    "isaac_physx_lidar": "isaac_physx_lidar",
    "usd_raycast": "custom_usd_raycast_laserscan",
    "custom_usd_raycast_laserscan": "custom_usd_raycast_laserscan",
}

CAPABILITY_MODULES = [
    "omni.kit.commands",
    "omni.isaac.sensor",
    "omni.isaac.range_sensor",
    "isaacsim.sensors.rtx",
    "omni.replicator.core",
    "omni.physx",
    "pxr.Usd",
    "pxr.UsdGeom",
    "pxr.Gf",
]

RTX_ANNOTATOR_CANDIDATES = [
    "IsaacCreateRTXLidarScanBufferForFlatScan",
    "IsaacCreateRTXLidarScanBuffer",
    "IsaacComputeRTXLidarFlatScan",
    "IsaacExtractRTXSensorPointCloudNoAccumulator",
    "RtxSensorCpuIsaacCreateRTXLidarScanBuffer",
    "RtxSensorCpuIsaacComputeRTXLidarPointCloud",
    "RtxSensorGpuIsaacCreateRTXLidarScanBuffer",
    "RtxSensorGpuIsaacComputeRTXLidarPointCloud",
]


class RealLidarUnavailable(RuntimeError):
    """Raised when a requested real LiDAR backend cannot be used."""


@dataclass(frozen=True)
class RealLidarConfig:
    backend: str = "auto"
    enable_laserscan_2d: bool = False
    enable_lidar_3d: bool = False
    frame_id: str = "laser"
    parent_frame_id: str = "base_link"
    mount_height_m: float = 0.25
    yaw_offset_rad: float = 0.0
    angle_min: float = -math.pi
    angle_max: float = math.pi
    angle_increment: float = 0.008726646
    range_min: float = 0.10
    range_max: float = 20.0
    vertical_fov_deg: float = 30.0
    horizontal_resolution_deg: float = 0.5
    vertical_resolution_deg: float = 1.0

    @property
    def beam_count(self) -> int:
        if self.angle_increment <= 0.0:
            raise ValueError("scan angle_increment must be positive")
        if self.angle_max <= self.angle_min:
            raise ValueError("scan angle_max must be greater than angle_min")
        return int(math.floor((self.angle_max - self.angle_min) / self.angle_increment)) + 1


def _import_status(module_name: str) -> dict[str, Any]:
    try:
        importlib.import_module(module_name)
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"available": True, "error": None}


def _probe_rtx_annotators() -> dict[str, Any]:
    status = _import_status("omni.replicator.core")
    if not status["available"]:
        return {"available": False, "annotators": {}, "error": status["error"]}
    try:
        import omni.replicator.core as rep  # type: ignore
    except Exception as exc:
        return {"available": False, "annotators": {}, "error": f"{type(exc).__name__}: {exc}"}

    annotators: dict[str, dict[str, Any]] = {}
    for name in RTX_ANNOTATOR_CANDIDATES:
        try:
            try:
                rep.AnnotatorRegistry.get_annotator(name, device="cpu")
            except TypeError:
                rep.AnnotatorRegistry.get_annotator(name)
            annotators[name] = {"available": True, "error": None}
        except Exception as exc:
            annotators[name] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "available": any(item["available"] for item in annotators.values()),
        "annotators": annotators,
        "error": None,
    }


def check_lidar_capabilities(*, isaac_python: str | None = None) -> dict[str, Any]:
    """Return a structured view of LiDAR APIs visible to this interpreter."""

    modules = {name: _import_status(name) for name in CAPABILITY_MODULES}
    rtx_annotators = _probe_rtx_annotators()

    kit_commands = modules["omni.kit.commands"]["available"]
    rtx_available = bool(kit_commands and (modules["isaacsim.sensors.rtx"]["available"] or rtx_annotators["available"]))
    range_available = bool(kit_commands and modules["omni.isaac.range_sensor"]["available"])
    physx_available = bool(modules["omni.physx"]["available"])
    usd_raycast_available = bool(
        modules["pxr.Usd"]["available"] and modules["pxr.UsdGeom"]["available"] and modules["pxr.Gf"]["available"]
    )

    backend_status = {
        "isaac_rtx_lidar": {
            "available": rtx_available,
            "required": ["omni.kit.commands", "isaacsim.sensors.rtx or RTX LiDAR annotator", "omni.replicator.core"],
            "notes": "RTX LiDAR collection uses an Isaac sensor prim plus RTX LiDAR annotator output.",
        },
        "isaac_range_sensor_lidar": {
            "available": range_available,
            "required": ["omni.kit.commands", "omni.isaac.range_sensor"],
            "notes": "RangeSensor LiDAR collection uses RangeSensorCreateLidar and the range sensor interface.",
        },
        "isaac_physx_lidar": {
            "available": physx_available,
            "required": ["omni.physx"],
            "notes": "PhysX raycast LiDAR traces the loaded Isaac USD scene through the PhysX scene query interface.",
        },
        "custom_usd_raycast_laserscan": {
            "available": usd_raycast_available,
            "required": ["pxr.Usd", "pxr.UsdGeom", "pxr.Gf"],
            "notes": "Explicit fallback only; traces loaded USD mesh geometry and is not RTX LiDAR.",
        },
    }
    available_backends = [name for name in ALL_BACKENDS if backend_status[name]["available"]]
    selected = next((name for name in ALL_BACKENDS if backend_status[name]["available"]), None)
    notes: list[str] = []
    if not any(backend_status[name]["available"] for name in TRUE_ISAAC_BACKENDS):
        notes.append("No true Isaac RTX/RangeSensor/PhysX LiDAR backend is currently visible to this interpreter.")
    if usd_raycast_available:
        notes.append("USD raycast fallback is available only with --lidar-backend usd_raycast; auto mode does not select it.")
    else:
        notes.append("USD raycast fallback is unavailable because one or more pxr USD modules are missing.")

    return {
        "available_backends": available_backends,
        "backend_status": backend_status,
        "can_collect_real_laserscan": bool(available_backends),
        "can_collect_real_lidar_3d": bool(rtx_available or range_available or usd_raycast_available),
        "isaac_python": isaac_python or sys.executable,
        "modules": modules,
        "notes": notes,
        "rtx_annotators": rtx_annotators,
        "selected_recommended_backend": selected,
    }


def write_lidar_capability_report(out: str | Path, capabilities: dict[str, Any]) -> dict[str, Path]:
    root = ensure_dir(out)
    json_path = write_json(root / "isaac_lidar_capabilities.json", capabilities)
    lines = [
        "# Isaac LiDAR Capabilities",
        "",
        f"- Isaac Python: `{capabilities.get('isaac_python')}`",
        f"- Available backends: {', '.join(capabilities.get('available_backends') or []) or 'none'}",
        f"- Selected recommended backend: `{capabilities.get('selected_recommended_backend')}`",
        f"- Can collect real LaserScan: `{bool(capabilities.get('can_collect_real_laserscan'))}`",
        f"- Can collect real 3D LiDAR: `{bool(capabilities.get('can_collect_real_lidar_3d'))}`",
        "",
        "## Backend Status",
    ]
    for name, status in capabilities.get("backend_status", {}).items():
        lines.append(f"- `{name}`: available={bool(status.get('available'))}; {status.get('notes')}")
    lines.extend(["", "## Notes"])
    for note in capabilities.get("notes", []):
        lines.append(f"- {note}")
    lines.extend(["", "## Module Imports"])
    for name, status in capabilities.get("modules", {}).items():
        if status.get("available"):
            lines.append(f"- `{name}`: ok")
        else:
            lines.append(f"- `{name}`: missing ({status.get('error')})")
    md_path = root / "isaac_lidar_capabilities_summary.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": json_path, "summary": md_path}


def normalize_requested_backend(value: str) -> str:
    key = str(value or "auto").strip().lower()
    if key not in BACKEND_ALIASES:
        raise ValueError(f"Unsupported LiDAR backend {value!r}; expected one of {sorted(BACKEND_ALIASES)}")
    return BACKEND_ALIASES[key]


def select_real_lidar_backend(requested: str, capabilities: dict[str, Any]) -> str:
    backend = normalize_requested_backend(requested)
    status = capabilities.get("backend_status", {})
    if backend == "auto":
        for name in TRUE_ISAAC_BACKENDS:
            if status.get(name, {}).get("available"):
                return name
        raise RealLidarUnavailable(
            "No true Isaac LiDAR backend is available for --lidar-backend auto. "
            "Run scripts/check_isaac_lidar_capabilities.py for details; "
            "use --lidar-backend usd_raycast only as an explicit geometry raycast fallback."
        )
    if backend == "custom_usd_raycast_laserscan" and not status.get(backend, {}).get("available"):
        raise RealLidarUnavailable("USD raycast fallback was requested but pxr USD geometry APIs are unavailable.")
    if not status.get(backend, {}).get("available"):
        raise RealLidarUnavailable(f"Requested LiDAR backend is unavailable: {backend}")
    return backend


def _base_pose(row: dict[str, Any]) -> tuple[float, float, float]:
    pose = row.get("base_pose_world")
    if not isinstance(pose, list) or len(pose) != 3:
        raise ValueError(f"frame row missing base_pose_world=[x,y,yaw]: {row!r}")
    return float(pose[0]), float(pose[1]), float(pose[2])


def laser_pose_base_link(config: RealLidarConfig) -> dict[str, Any]:
    return {
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": float(config.yaw_offset_rad),
        "x": 0.0,
        "y": 0.0,
        "z": float(config.mount_height_m),
    }


def laser_pose_world(row: dict[str, Any], config: RealLidarConfig) -> tuple[np.ndarray, float]:
    x, y, yaw = _base_pose(row)
    return np.asarray([x, y, float(config.mount_height_m)], dtype=np.float64), float(yaw + config.yaw_offset_rad)


def _scan_angles(config: RealLidarConfig) -> np.ndarray:
    return np.asarray([config.angle_min + i * config.angle_increment for i in range(config.beam_count)], dtype=np.float64)


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    return value


class UsdMeshRaycaster:
    """Small USD mesh raycaster used only for explicit geometry fallback."""

    def __init__(self, scene_usd: str | Path, *, z_filter: float | None = None) -> None:
        try:
            from pxr import Usd, UsdGeom  # type: ignore
        except Exception as exc:
            raise RealLidarUnavailable(f"pxr USD modules are unavailable for USD raycast fallback: {type(exc).__name__}: {exc}") from exc

        stage = Usd.Stage.Open(str(scene_usd))
        if stage is None:
            raise FileNotFoundError(f"USD stage could not be opened for raycast fallback: {scene_usd}")

        triangles: list[np.ndarray] = []
        mesh_count = 0
        for prim in stage.Traverse():
            if not prim.IsA(UsdGeom.Mesh):
                continue
            mesh = UsdGeom.Mesh(prim)
            points_attr = mesh.GetPointsAttr().Get()
            counts_attr = mesh.GetFaceVertexCountsAttr().Get()
            indices_attr = mesh.GetFaceVertexIndicesAttr().Get()
            if points_attr is None or counts_attr is None or indices_attr is None:
                continue
            points = np.asarray([[float(p[0]), float(p[1]), float(p[2]), 1.0] for p in points_attr], dtype=np.float64)
            if points.size == 0:
                continue
            matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            transform = np.asarray([[float(matrix[i][j]) for j in range(4)] for i in range(4)], dtype=np.float64)
            world_points = (points @ transform.T)[:, :3]
            cursor = 0
            for count in counts_attr:
                count_i = int(count)
                face = [int(v) for v in indices_attr[cursor : cursor + count_i]]
                cursor += count_i
                if count_i < 3:
                    continue
                for tri_idx in range(1, count_i - 1):
                    tri = world_points[[face[0], face[tri_idx], face[tri_idx + 1]], :]
                    if z_filter is not None:
                        min_z = float(np.min(tri[:, 2]))
                        max_z = float(np.max(tri[:, 2]))
                        if z_filter < min_z or z_filter > max_z:
                            continue
                    triangles.append(tri.astype(np.float64))
            mesh_count += 1
        if not triangles:
            raise RealLidarUnavailable(f"No USD mesh triangles were available for raycast fallback: {scene_usd}")
        self.mesh_count = mesh_count
        self.triangles = np.stack(triangles, axis=0)

    def raycast(self, origin: np.ndarray, direction: np.ndarray, max_distance: float) -> float | None:
        triangles = self.triangles
        eps = 1e-8
        v0 = triangles[:, 0, :]
        edge1 = triangles[:, 1, :] - v0
        edge2 = triangles[:, 2, :] - v0
        pvec = np.cross(np.broadcast_to(direction, edge2.shape), edge2)
        det = np.einsum("ij,ij->i", edge1, pvec)
        det_mask = np.abs(det) > eps
        if not np.any(det_mask):
            return None
        inv_det = np.zeros_like(det)
        inv_det[det_mask] = 1.0 / det[det_mask]
        tvec = origin.reshape(1, 3) - v0
        u = np.einsum("ij,ij->i", tvec, pvec) * inv_det
        qvec = np.cross(tvec, edge1)
        v = np.einsum("j,ij->i", direction, qvec) * inv_det
        t = np.einsum("ij,ij->i", edge2, qvec) * inv_det
        mask = det_mask & (u >= 0.0) & (v >= 0.0) & ((u + v) <= 1.0) & (t >= 0.0) & (t <= float(max_distance))
        if not np.any(mask):
            return None
        return float(np.min(t[mask]))


def _make_scan_record(
    *,
    row: dict[str, Any],
    frame_index: int,
    ranges: Iterable[float],
    backend: str,
    config: RealLidarConfig,
    scan_quality: str,
) -> dict[str, Any]:
    timestamp = float(row.get("timestamp", row.get("t", frame_index)))
    return {
        "angle_increment": float(config.angle_increment),
        "angle_max": float(config.angle_max),
        "angle_min": float(config.angle_min),
        "backend": backend,
        "frame_id": config.frame_id,
        "frame_index": int(row.get("frame_idx", frame_index)),
        "intensities": [],
        "is_depth_derived": False,
        "is_real_lidar": True,
        "parent_frame_id": config.parent_frame_id,
        "pose_base_link": laser_pose_base_link(config),
        "range_max": float(config.range_max),
        "range_min": float(config.range_min),
        "ranges": [float(v) for v in ranges],
        "scan_quality": scan_quality,
        "scan_time": 0.0,
        "time_increment": 0.0,
        "timestamp_sec": timestamp,
    }


def _save_scan_pair(scan_dir: Path, stem: str, scan: dict[str, Any]) -> tuple[str, str]:
    rel_json = f"sensors/laserscan_2d/{stem}.json"
    rel_npy = f"sensors/laserscan_2d/{stem}.npy"
    save_laserscan(scan_dir.parent.parent / rel_json, scan)
    save_laserscan_npy(scan_dir.parent.parent / rel_npy, scan)
    return rel_json, rel_npy


def _save_lidar_npz(path: Path, points_xyz: np.ndarray, *, frame_index: int, timestamp: float, backend: str, frame_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "backend": backend,
        "frame_id": frame_id,
        "frame_index": int(frame_index),
        "is_depth_derived": False,
        "is_real_lidar": True,
        "timestamp_sec": float(timestamp),
    }
    np.savez_compressed(
        path,
        intensity=np.zeros((points_xyz.shape[0],), dtype=np.float32),
        metadata_json=json.dumps(metadata, sort_keys=True),
        points_xyz=np.asarray(points_xyz, dtype=np.float32),
    )


def _annotate_row_real_lidar(row: dict[str, Any], *, backend: str, config: RealLidarConfig, scan_source: str, scan_quality: str) -> None:
    row["depth_derived_scan"] = False
    row["is_depth_derived"] = False
    row["is_real_lidar"] = True
    row["lidar_backend"] = backend
    row["lidar_frame_id"] = config.frame_id
    row["lidar_parent_frame_id"] = config.parent_frame_id
    row["lidar_pose_base_link"] = laser_pose_base_link(config)
    row["scan_quality"] = scan_quality
    row["scan_source"] = scan_source


def _collect_usd_raycast(
    *,
    dataset: Path,
    scene_usd: str | Path,
    manifest_rows: list[dict[str, Any]],
    config: RealLidarConfig,
) -> dict[str, Any]:
    raycaster = UsdMeshRaycaster(scene_usd, z_filter=float(config.mount_height_m) if config.enable_laserscan_2d else None)
    scan_dir = ensure_dir(dataset / "sensors" / "laserscan_2d")
    lidar_dir = ensure_dir(dataset / "sensors" / "lidar_3d")
    angles = _scan_angles(config)
    scan_stats_rows: list[dict[str, Any]] = []
    scan_count = 0
    lidar_count = 0
    scan_quality = "geometry_raycast_fallback_not_rtx_lidar"

    for local_idx, row in enumerate(manifest_rows):
        stem = f"{int(row.get('frame_idx', local_idx)):06d}"
        origin, yaw = laser_pose_world(row, config)
        ranges = np.full((angles.size,), float(config.range_max), dtype=np.float32)
        if config.enable_laserscan_2d:
            for idx, angle in enumerate(angles):
                world_angle = yaw + float(angle)
                direction = np.asarray([math.cos(world_angle), math.sin(world_angle), 0.0], dtype=np.float64)
                distance = raycaster.raycast(origin, direction, float(config.range_max))
                if distance is not None and distance >= float(config.range_min):
                    ranges[idx] = float(distance)
            scan = _make_scan_record(
                row=row,
                frame_index=local_idx,
                ranges=ranges,
                backend="usd_raycast",
                config=config,
                scan_quality=scan_quality,
            )
            rel_json, rel_npy = _save_scan_pair(scan_dir, stem, scan)
            row["laserscan_2d_path"] = rel_json
            row["laserscan_2d_ranges_path"] = rel_npy
            _annotate_row_real_lidar(
                row,
                backend="usd_raycast",
                config=config,
                scan_source="usd_raycast_laserscan_2d",
                scan_quality=scan_quality,
            )
            scan_stats_rows.append({"frame_idx": int(row.get("frame_idx", local_idx)), **laserscan_stats(scan)})
            scan_count += 1
        if config.enable_lidar_3d:
            points_3d: list[np.ndarray] = []
            horiz_step = math.radians(max(1e-6, float(config.horizontal_resolution_deg)))
            horizontal_angles = np.arange(float(config.angle_min), float(config.angle_max) + 0.5 * horiz_step, horiz_step, dtype=np.float64)
            vertical_step = math.radians(max(1e-6, float(config.vertical_resolution_deg)))
            half_v = math.radians(float(config.vertical_fov_deg) * 0.5)
            vertical_angles = np.arange(-half_v, half_v + 0.5 * vertical_step, vertical_step, dtype=np.float64)
            for elev in vertical_angles:
                cos_elev = math.cos(float(elev))
                sin_elev = math.sin(float(elev))
                for angle in horizontal_angles:
                    world_angle = yaw + float(angle)
                    direction = np.asarray([cos_elev * math.cos(world_angle), cos_elev * math.sin(world_angle), sin_elev], dtype=np.float64)
                    distance = raycaster.raycast(origin, direction, float(config.range_max))
                    if distance is not None and distance >= float(config.range_min):
                        points_3d.append(origin + direction * float(distance))
            points = np.asarray(points_3d, dtype=np.float32).reshape((-1, 3)) if points_3d else np.empty((0, 3), dtype=np.float32)
            timestamp = float(row.get("timestamp", row.get("t", local_idx)))
            rel_npz = f"sensors/lidar_3d/{stem}.npz"
            _save_lidar_npz(dataset / rel_npz, points, frame_index=int(row.get("frame_idx", local_idx)), timestamp=timestamp, backend="usd_raycast", frame_id=config.frame_id)
            row["lidar_3d_path"] = rel_npz
            if not config.enable_laserscan_2d:
                _annotate_row_real_lidar(
                    row,
                    backend="usd_raycast",
                    config=config,
                    scan_source="usd_raycast_lidar_3d_projected",
                    scan_quality=scan_quality,
                )
            lidar_count += 1

    return {
        "backend": "usd_raycast",
        "is_depth_derived": False,
        "is_real_lidar": True,
        "lidar_3d_count": lidar_count,
        "mesh_count": raycaster.mesh_count,
        "scan_count": scan_count,
        "scan_quality": scan_quality,
        "scan_source": "usd_raycast_laserscan_2d" if scan_count else "usd_raycast_lidar_3d_projected",
        "scan_stats_sample": scan_stats_rows[:20],
        "triangle_count": int(raycaster.triangles.shape[0]),
    }


def _scan_from_range_values(values: Any, *, row: dict[str, Any], frame_index: int, backend: str, config: RealLidarConfig, quality: str) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        arr = arr[arr.shape[0] // 2, :]
    arr = arr.reshape(-1)
    if arr.size == 0:
        raise RuntimeError(f"{backend} returned an empty range array")
    if arr.size != config.beam_count:
        x_old = np.linspace(0.0, 1.0, num=arr.size, dtype=np.float64)
        x_new = np.linspace(0.0, 1.0, num=config.beam_count, dtype=np.float64)
        arr = np.interp(x_new, x_old, arr.astype(np.float64)).astype(np.float32)
    ranges = np.where(np.isfinite(arr) & (arr >= config.range_min) & (arr <= config.range_max), arr, config.range_max)
    return _make_scan_record(row=row, frame_index=frame_index, ranges=ranges, backend=backend, config=config, scan_quality=quality)


def _extract_points_from_any(data: Any) -> np.ndarray | None:
    if data is None:
        return None
    if isinstance(data, dict):
        for key in ("points", "pointCloud", "point_cloud", "pointcloud", "data"):
            if key in data:
                pts = _extract_points_from_any(data[key])
                if pts is not None:
                    return pts
        return None
    arr = np.asarray(data)
    if arr.size == 0:
        return None
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1 and arr.size % 3 == 0:
        arr = arr.reshape((-1, 3))
    if arr.ndim == 2 and arr.shape[1] >= 3:
        return arr[:, :3].astype(np.float32)
    return None


def _extract_ranges_from_any(data: Any) -> np.ndarray | None:
    if data is None:
        return None
    if isinstance(data, dict):
        for key in ("ranges", "range", "rangesData", "linearDepthData", "depth", "data"):
            if key in data:
                ranges = _extract_ranges_from_any(data[key])
                if ranges is not None:
                    return ranges
        return None
    arr = np.asarray(data)
    if arr.size == 0:
        return None
    if arr.ndim <= 2 and arr.size >= 2:
        return np.asarray(arr, dtype=np.float32)
    return None


def _import_simulation_app() -> Any:
    try:
        from isaacsim import SimulationApp  # type: ignore

        return SimulationApp
    except Exception:
        from omni.isaac.kit import SimulationApp  # type: ignore

        return SimulationApp


def _open_isaac_world(scene_usd: str | Path, *, headless: bool) -> tuple[Any, Any]:
    SimulationApp = _import_simulation_app()
    app = SimulationApp({"headless": bool(headless)})
    try:
        try:
            from isaacsim.core.api import World  # type: ignore
            from isaacsim.core.utils.stage import define_prim, open_stage  # type: ignore
        except Exception:
            from omni.isaac.core import World  # type: ignore
            from omni.isaac.core.utils.stage import define_prim, open_stage  # type: ignore

        scene_loaded = open_stage(str(scene_usd))
        if scene_loaded is False:
            raise RuntimeError(f"Isaac Sim failed to open scene USD: {scene_usd}")
        world = World(stage_units_in_meters=1.0)
        world.reset()
        if hasattr(world, "play"):
            world.play()
        define_prim("/World/OracleReplayRobot", "Xform")
        return app, world
    except Exception:
        app.close()
        raise


def _set_usd_xform_pose(prim_path: str, row: dict[str, Any]) -> None:
    from pxr import Gf, UsdGeom  # type: ignore
    import omni.usd  # type: ignore

    x, y, yaw = _base_pose(row)
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    matrix = Gf.Matrix4d(1.0)
    matrix.SetRotate(Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), math.degrees(yaw)))
    matrix.SetTranslateOnly(Gf.Vec3d(x, y, 0.0))
    xformable.AddTransformOp().Set(matrix)


def _set_lidar_local_xform(prim_path: str, config: RealLidarConfig) -> None:
    from pxr import Gf, UsdGeom  # type: ignore
    import omni.usd  # type: ignore

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"LiDAR prim does not exist: {prim_path}")
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    matrix = Gf.Matrix4d(1.0)
    matrix.SetRotate(Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), math.degrees(config.yaw_offset_rad)))
    matrix.SetTranslateOnly(Gf.Vec3d(0.0, 0.0, float(config.mount_height_m)))
    xformable.AddTransformOp().Set(matrix)


def _created_prim_from_command_result(result: Any) -> Any | None:
    values = result if isinstance(result, tuple) else (result,)
    for value in reversed(values):
        if hasattr(value, "IsValid") and value.IsValid():
            return value
    return None


def _rtx_lidar_config_name(config: RealLidarConfig) -> str:
    return "RPLIDAR_S2E" if config.enable_laserscan_2d else "Example_Rotary"


def _yaw_quat_gf(yaw: float) -> Any:
    from pxr import Gf  # type: ignore

    half = float(yaw) * 0.5
    return Gf.Quatd(math.cos(half), 0.0, 0.0, math.sin(half))


def _run_async(coro: Any) -> Any:
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    if loop.is_running():
        raise RuntimeError("Cannot wait for Isaac sensor updates while the asyncio event loop is already running.")
    return loop.run_until_complete(coro)


def _advance_isaac_sensor_frame(world: Any, *, updates: int = 8) -> None:
    try:
        import omni.kit.app  # type: ignore

        app = omni.kit.app.get_app()
    except Exception:
        app = None
    for _ in range(max(1, int(updates))):
        world.step(render=True)
        if app is not None:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    continue
                _run_async(app.next_update_async())
            except Exception:
                pass


def _collect_range_sensor(
    *,
    dataset: Path,
    scene_usd: str | Path,
    manifest_rows: list[dict[str, Any]],
    config: RealLidarConfig,
    headless: bool,
    world: Any | None = None,
) -> dict[str, Any]:
    app = None
    if world is None:
        app, world = _open_isaac_world(scene_usd, headless=headless)
    try:
        import omni.kit.commands  # type: ignore
        from omni.isaac.range_sensor import _range_sensor  # type: ignore

        parent = "/World/OracleReplayRobot"
        prim_path = f"{parent}/{config.frame_id}"
        failures: list[str] = []
        for kwargs in (
            {
                "path": prim_path,
                "parent": parent,
                "min_range": float(config.range_min),
                "max_range": float(config.range_max),
                "horizontal_fov": math.degrees(config.angle_max - config.angle_min),
                "vertical_fov": float(config.vertical_fov_deg),
                "horizontal_resolution": float(config.horizontal_resolution_deg),
                "vertical_resolution": float(config.vertical_resolution_deg),
                "rotation_rate": 0.0,
                "draw_points": False,
                "draw_lines": False,
            },
            {
                "path": prim_path,
                "parent": parent,
                "min_range": float(config.range_min),
                "max_range": float(config.range_max),
            },
        ):
            try:
                omni.kit.commands.execute("RangeSensorCreateLidar", **kwargs)
                failures = []
                break
            except Exception as exc:
                failures.append(f"{type(exc).__name__}: {exc}")
        if failures:
            raise RealLidarUnavailable("RangeSensorCreateLidar failed: " + "; ".join(failures))
        _set_lidar_local_xform(prim_path, config)
        interface = _range_sensor.acquire_lidar_sensor_interface()
        scan_dir = ensure_dir(dataset / "sensors" / "laserscan_2d")
        lidar_dir = ensure_dir(dataset / "sensors" / "lidar_3d")
        scan_count = 0
        lidar_count = 0
        for local_idx, row in enumerate(manifest_rows):
            stem = f"{int(row.get('frame_idx', local_idx)):06d}"
            _set_usd_xform_pose(parent, row)
            for _ in range(3):
                world.step(render=True)
            if config.enable_laserscan_2d:
                ranges = None
                for method_name in ("get_linear_depth_data", "get_depth_data"):
                    if hasattr(interface, method_name):
                        try:
                            ranges = getattr(interface, method_name)(prim_path)
                            break
                        except Exception:
                            continue
                scan = _scan_from_range_values(ranges, row=row, frame_index=local_idx, backend="isaac_range_sensor_lidar", config=config, quality="real_isaac_lidar")
                rel_json, rel_npy = _save_scan_pair(scan_dir, stem, scan)
                row["laserscan_2d_path"] = rel_json
                row["laserscan_2d_ranges_path"] = rel_npy
                scan_count += 1
            if config.enable_lidar_3d:
                points = None
                if hasattr(interface, "get_point_cloud_data"):
                    points = _extract_points_from_any(interface.get_point_cloud_data(prim_path))
                if points is None:
                    raise RuntimeError("RangeSensor LiDAR did not expose point cloud data for --enable-real-3d-lidar")
                rel_npz = f"sensors/lidar_3d/{stem}.npz"
                _save_lidar_npz(
                    dataset / rel_npz,
                    points,
                    frame_index=int(row.get("frame_idx", local_idx)),
                    timestamp=float(row.get("timestamp", row.get("t", local_idx))),
                    backend="isaac_range_sensor_lidar",
                    frame_id=config.frame_id,
                )
                row["lidar_3d_path"] = rel_npz
                lidar_count += 1
            _annotate_row_real_lidar(
                row,
                backend="isaac_range_sensor_lidar",
                config=config,
                scan_source="isaac_laserscan_2d" if config.enable_laserscan_2d else "isaac_lidar_3d_projected",
                scan_quality="real_isaac_lidar",
            )
        return {
            "backend": "isaac_range_sensor_lidar",
            "is_depth_derived": False,
            "is_real_lidar": True,
            "lidar_3d_count": lidar_count,
            "prim_path": prim_path,
            "scan_count": scan_count,
            "scan_quality": "real_isaac_lidar",
            "scan_source": "isaac_laserscan_2d" if scan_count else "isaac_lidar_3d_projected",
        }
    finally:
        if app is not None:
            app.close()


def _collect_rtx_lidar(
    *,
    dataset: Path,
    scene_usd: str | Path,
    manifest_rows: list[dict[str, Any]],
    config: RealLidarConfig,
    headless: bool,
    world: Any | None = None,
) -> dict[str, Any]:
    app = None
    if world is None:
        app, world = _open_isaac_world(scene_usd, headless=headless)
    try:
        import omni.kit.commands  # type: ignore
        import omni.replicator.core as rep  # type: ignore
        from pxr import Gf  # type: ignore

        parent = "/World/OracleReplayRobot"
        prim_path = f"{parent}/{config.frame_id}"
        failures: list[str] = []
        sensor_prim = None
        config_name = _rtx_lidar_config_name(config)
        for kwargs in (
            {
                "path": prim_path,
                "config": config_name,
                "translation": Gf.Vec3d(0.0, 0.0, float(config.mount_height_m)),
                "orientation": _yaw_quat_gf(config.yaw_offset_rad),
                "omni:sensor:Core:outputFrameOfReference": "WORLD",
                "omni:sensor:Core:auxOutputType": "FULL",
            },
            {
                "path": config.frame_id,
                "parent": parent,
                "config": config_name,
                "translation": Gf.Vec3d(0.0, 0.0, float(config.mount_height_m)),
                "orientation": _yaw_quat_gf(config.yaw_offset_rad),
                "omni:sensor:Core:outputFrameOfReference": "WORLD",
                "omni:sensor:Core:auxOutputType": "FULL",
            },
            {
                "path": prim_path,
                "config": "Example_Rotary",
                "translation": Gf.Vec3d(0.0, 0.0, float(config.mount_height_m)),
                "orientation": _yaw_quat_gf(config.yaw_offset_rad),
            },
            {"path": prim_path, "parent": parent},
        ):
            try:
                result = omni.kit.commands.execute("IsaacSensorCreateRtxLidar", **kwargs)
                sensor_prim = _created_prim_from_command_result(result)
                if sensor_prim is None:
                    raise RuntimeError(f"command returned no valid prim: {result!r}")
                failures = []
                prim_path = str(sensor_prim.GetPath())
                break
            except Exception as exc:
                failures.append(f"{type(exc).__name__}: {exc}")
        if failures:
            raise RealLidarUnavailable("IsaacSensorCreateRtxLidar failed: " + "; ".join(failures))
        _set_lidar_local_xform(prim_path, config)
        render_product = rep.create.render_product(
            sensor_prim.GetPath() if sensor_prim is not None else prim_path,
            [32, 32],
            name="OracleRtxLidarRenderProduct",
            render_vars=["GenericModelOutput", "RtxSensorMetadata"],
        )
        render_product_target = getattr(render_product, "path", render_product)
        annotators = []
        for name in RTX_ANNOTATOR_CANDIDATES:
            try:
                try:
                    annotator = rep.AnnotatorRegistry.get_annotator(name, device="cpu")
                except TypeError:
                    annotator = rep.AnnotatorRegistry.get_annotator(name)
                if name == "IsaacCreateRTXLidarScanBuffer":
                    try:
                        annotator.initialize(outputDistance=True, outputIntensity=True, outputAzimuth=True, outputElevation=True)
                    except Exception:
                        pass
                annotator.attach([render_product_target])
                annotators.append((name, annotator))
            except Exception:
                continue
        if not annotators:
            raise RealLidarUnavailable("No RTX LiDAR annotator could be attached to the render product.")
        try:
            import omni.timeline  # type: ignore

            omni.timeline.get_timeline_interface().play()
        except Exception:
            pass
        _advance_isaac_sensor_frame(world, updates=10)
        scan_dir = ensure_dir(dataset / "sensors" / "laserscan_2d")
        lidar_dir = ensure_dir(dataset / "sensors" / "lidar_3d")
        scan_count = 0
        lidar_count = 0
        for local_idx, row in enumerate(manifest_rows):
            stem = f"{int(row.get('frame_idx', local_idx)):06d}"
            _set_usd_xform_pose(parent, row)
            _advance_isaac_sensor_frame(world, updates=10)
            data_by_name = {}
            for name, annotator in annotators:
                try:
                    data_by_name[name] = annotator.get_data()
                except Exception as exc:
                    data_by_name[name] = {"error": f"{type(exc).__name__}: {exc}"}
            points = next((pts for pts in (_extract_points_from_any(data) for data in data_by_name.values()) if pts is not None), None)
            ranges = next((rng for rng in (_extract_ranges_from_any(data) for data in data_by_name.values()) if rng is not None), None)
            if config.enable_laserscan_2d:
                if ranges is not None:
                    scan = _scan_from_range_values(ranges, row=row, frame_index=local_idx, backend="isaac_rtx_lidar", config=config, quality="real_isaac_lidar")
                elif points is not None:
                    scan = pointcloud_to_laserscan(
                        points,
                        angle_min=config.angle_min,
                        angle_max=config.angle_max,
                        angle_increment=config.angle_increment,
                        range_min=config.range_min,
                        range_max=config.range_max,
                        frame_id=config.frame_id,
                    )
                    scan.update(
                        {
                            "backend": "isaac_rtx_lidar",
                            "frame_index": int(row.get("frame_idx", local_idx)),
                            "is_depth_derived": False,
                            "is_real_lidar": True,
                            "parent_frame_id": config.parent_frame_id,
                            "pose_base_link": laser_pose_base_link(config),
                            "scan_quality": "real_isaac_lidar",
                            "timestamp_sec": float(row.get("timestamp", row.get("t", local_idx))),
                        }
                    )
                else:
                    raise RuntimeError(f"RTX LiDAR annotators returned no ranges or points; data keys={list(data_by_name)}")
                rel_json, rel_npy = _save_scan_pair(scan_dir, stem, scan)
                row["laserscan_2d_path"] = rel_json
                row["laserscan_2d_ranges_path"] = rel_npy
                scan_count += 1
            if config.enable_lidar_3d:
                if points is None:
                    raise RuntimeError("RTX LiDAR did not expose point cloud data for --enable-real-3d-lidar")
                rel_npz = f"sensors/lidar_3d/{stem}.npz"
                _save_lidar_npz(
                    dataset / rel_npz,
                    points,
                    frame_index=int(row.get("frame_idx", local_idx)),
                    timestamp=float(row.get("timestamp", row.get("t", local_idx))),
                    backend="isaac_rtx_lidar",
                    frame_id=config.frame_id,
                )
                row["lidar_3d_path"] = rel_npz
                lidar_count += 1
            _annotate_row_real_lidar(
                row,
                backend="isaac_rtx_lidar",
                config=config,
                scan_source="isaac_laserscan_2d" if config.enable_laserscan_2d else "isaac_lidar_3d_projected",
                scan_quality="real_isaac_lidar",
            )
        return {
            "annotators": [name for name, _ in annotators],
            "backend": "isaac_rtx_lidar",
            "config": config_name,
            "is_depth_derived": False,
            "is_real_lidar": True,
            "lidar_3d_count": lidar_count,
            "prim_path": prim_path,
            "scan_count": scan_count,
            "scan_quality": "real_isaac_lidar",
            "scan_source": "isaac_laserscan_2d" if scan_count else "isaac_lidar_3d_projected",
        }
    finally:
        if app is not None:
            app.close()


def _physx_hit_distance(hit: Any) -> float | None:
    if hit is None:
        return None
    if isinstance(hit, dict):
        if hit.get("hit") is False:
            return None
        for key in ("distance", "dist", "range"):
            if key in hit:
                value = float(hit[key])
                return value if math.isfinite(value) else None
        return None
    if hasattr(hit, "hit") and not bool(hit.hit):
        return None
    for key in ("distance", "dist"):
        if hasattr(hit, key):
            value = float(getattr(hit, key))
            return value if math.isfinite(value) else None
    return None


def _collect_physx_raycast(
    *,
    dataset: Path,
    scene_usd: str | Path,
    manifest_rows: list[dict[str, Any]],
    config: RealLidarConfig,
    headless: bool,
    world: Any | None = None,
) -> dict[str, Any]:
    app = None
    if world is None:
        app, world = _open_isaac_world(scene_usd, headless=headless)
    try:
        import omni.physx  # type: ignore

        query = omni.physx.get_physx_scene_query_interface()
        scan_dir = ensure_dir(dataset / "sensors" / "laserscan_2d")
        angles = _scan_angles(config)
        scan_count = 0
        for local_idx, row in enumerate(manifest_rows):
            stem = f"{int(row.get('frame_idx', local_idx)):06d}"
            _set_usd_xform_pose("/World/OracleReplayRobot", row)
            for _ in range(2):
                world.step(render=True)
            origin, yaw = laser_pose_world(row, config)
            ranges = np.full((angles.size,), float(config.range_max), dtype=np.float32)
            for idx, angle in enumerate(angles):
                direction = np.asarray([math.cos(yaw + float(angle)), math.sin(yaw + float(angle)), 0.0], dtype=np.float32)
                hit = query.raycast_closest(tuple(origin.astype(float)), tuple(direction.astype(float)), float(config.range_max))
                distance = _physx_hit_distance(hit)
                if distance is not None and distance >= float(config.range_min):
                    ranges[idx] = float(distance)
            scan = _make_scan_record(
                row=row,
                frame_index=local_idx,
                ranges=ranges,
                backend="isaac_physx_lidar",
                config=config,
                scan_quality="real_isaac_lidar",
            )
            rel_json, rel_npy = _save_scan_pair(scan_dir, stem, scan)
            row["laserscan_2d_path"] = rel_json
            row["laserscan_2d_ranges_path"] = rel_npy
            _annotate_row_real_lidar(
                row,
                backend="isaac_physx_lidar",
                config=config,
                scan_source="isaac_laserscan_2d",
                scan_quality="real_isaac_lidar",
            )
            scan_count += 1
        return {
            "backend": "isaac_physx_lidar",
            "is_depth_derived": False,
            "is_real_lidar": True,
            "lidar_3d_count": 0,
            "scan_count": scan_count,
            "scan_quality": "real_isaac_lidar",
            "scan_source": "isaac_laserscan_2d",
        }
    finally:
        if app is not None:
            app.close()


def collect_real_lidar_frames(
    *,
    dataset: str | Path,
    scene_usd: str | Path,
    manifest_rows: list[dict[str, Any]],
    config: RealLidarConfig,
    backend: str,
    headless: bool = True,
    world: Any | None = None,
) -> dict[str, Any]:
    dataset_path = Path(dataset)
    if not config.enable_laserscan_2d and not config.enable_lidar_3d:
        raise ValueError("At least one of enable_laserscan_2d or enable_lidar_3d must be true.")
    if backend == "custom_usd_raycast_laserscan":
        return _collect_usd_raycast(dataset=dataset_path, scene_usd=scene_usd, manifest_rows=manifest_rows, config=config)
    if backend == "isaac_range_sensor_lidar":
        return _collect_range_sensor(dataset=dataset_path, scene_usd=scene_usd, manifest_rows=manifest_rows, config=config, headless=headless, world=world)
    if backend == "isaac_rtx_lidar":
        return _collect_rtx_lidar(dataset=dataset_path, scene_usd=scene_usd, manifest_rows=manifest_rows, config=config, headless=headless, world=world)
    if backend == "isaac_physx_lidar":
        return _collect_physx_raycast(dataset=dataset_path, scene_usd=scene_usd, manifest_rows=manifest_rows, config=config, headless=headless, world=world)
    raise ValueError(f"Unsupported selected LiDAR backend: {backend}")


def real_lidar_metadata_update(config: RealLidarConfig, backend: str, collection: dict[str, Any]) -> dict[str, Any]:
    scan_count = int(collection.get("scan_count", 0))
    lidar_count = int(collection.get("lidar_3d_count", 0))
    scan_source = collection.get("scan_source")
    if not scan_source:
        if scan_count:
            scan_source = "isaac_laserscan_2d" if backend != "custom_usd_raycast_laserscan" else "usd_raycast_laserscan_2d"
        elif lidar_count:
            scan_source = "isaac_lidar_3d_projected" if backend != "custom_usd_raycast_laserscan" else "usd_raycast_lidar_3d_projected"
    return _json_ready(
        {
            "depth_derived_scan": False,
            "laserscan_2d_available": bool(scan_count),
            "lidar_3d_available": bool(lidar_count),
            "lidar_backend": collection.get("backend", backend),
            "lidar_backend_available": True,
            "lidar_collection": collection,
            "lidar_frame_id": config.frame_id,
            "lidar_mount_height_m": float(config.mount_height_m),
            "lidar_parent_frame_id": config.parent_frame_id,
            "real_lidar_enabled": True,
            "scan_quality": collection.get("scan_quality") or ("real_isaac_lidar" if backend != "custom_usd_raycast_laserscan" else "geometry_raycast_fallback_not_rtx_lidar"),
            "scan_source": scan_source,
        }
    )
