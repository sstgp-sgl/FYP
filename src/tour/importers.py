"""Import plain XYZ/RGB PLY files, old reconstruction results, Phase 1
combined.ply, and the deterministic synthetic demo scene.

Only backend-resolved paths are accepted — the browser can never pass an
arbitrary server path. Uploads are streamed with a byte cap and the PLY header
is inspected for the vertex count before parsing.
"""
from __future__ import annotations

import re
import tempfile
import uuid
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from .geometry import compute_display_transform, DEFAULT_COLOR
from .models import Camera, Entity, Fact, Shot, Storyboard, StoryboardValidation, TourScene, ViewPreset, now
from .repository import (
    ConflictError,
    InvalidDataError,
    NotFoundError,
    PayloadTooLargeError,
    TourRepository,
)

GAUSSIAN_MARKERS = ("f_dc_0", "opacity", "scale_0", "rot_0")

_SANITIZE_RE = re.compile(r"[^\w\u4e00-\u9fff-]+")


def sanitize_name(name: str) -> str:
    """Same rule as the old subject sanitizer (no import of src.subjects)."""
    name = _SANITIZE_RE.sub("_", (name or "subject").strip()).strip("_")
    return name or "subject"


def parse_ply_data(data: bytes, config: dict, source_label: str):
    """Parse PLY bytes with caps and Gaussian detection.

    Returns (points float64 Nx3, colors uint8 Nx3, colors_missing bool).
    Raises PayloadTooLargeError for the upload byte cap and the point cap.
    ``config`` is the resolved tour config dict (``{"tour": {...}, "llm": {...}}``).
    """
    max_points = int(config["tour"]["max_points"])
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as stream:
        stream.write(data)
        temp_path = Path(stream.name)

    try:
        header = _scan_header(temp_path)
        count = header.get("vertex_count")
        if count is not None and count > max_points:
            raise PayloadTooLargeError(
                f"{source_label} declares {count} vertices; limit is {max_points}"
            )
        try:
            ply = PlyData.read(str(temp_path))
        except Exception as exc:  # noqa: BLE001
            raise InvalidDataError(f"cannot parse PLY: {exc}") from exc
        vertex = ply["vertex"].data
        fields = set(vertex.dtype.names or ())
        if not {"x", "y", "z"} <= fields:
            raise InvalidDataError("PLY is missing x/y/z vertex properties")
        if len(vertex) == 0:
            raise InvalidDataError("PLY contains no vertices")
        if len(vertex) > max_points:
            raise PayloadTooLargeError(
                f"{source_label} contains {len(vertex)} vertices; limit is {max_points}"
            )
        if GAUSSIAN_MARKERS[0] in fields or {"opacity", "scale_0", "rot_0"} & fields:
            raise InvalidDataError(
                "Gaussian splat PLY detected; the tour supports plain XYZ/RGB "
                "point clouds only and will not render splats as points"
            )
        xyz = np.column_stack([vertex[k] for k in ("x", "y", "z")]).astype(np.float64)
        if not np.isfinite(xyz).all():
            raise InvalidDataError("PLY coordinates contain NaN or infinite values")

        colors = None
        colors_missing = False
        if {"red", "green", "blue"} <= fields:
            rgb = np.column_stack([vertex[k] for k in ("red", "green", "blue")])
            if rgb.dtype.kind == "u" and rgb.dtype.itemsize == 1:
                colors = rgb
            elif rgb.dtype.kind == "f":
                scaled = np.clip(rgb, 0.0, 1.0) * 255.0
                colors = scaled.astype(np.uint8)
            else:
                colors_missing = True
        else:
            colors_missing = True
        if colors is None:
            colors = np.tile(np.array(DEFAULT_COLOR, dtype=np.uint8), (len(xyz), 1))
        return xyz, colors, colors_missing
    finally:
        try:
            temp_path.unlink()
        except OSError:
            pass


def _scan_header(path: Path) -> dict:
    """Read only the ASCII header of a PLY to find the declared vertex count."""
    with open(path, "rb") as stream:
        chunk = bytearray()
        while b"end_header" not in chunk and len(chunk) < 1 << 20:
            piece = stream.read(4096)
            if not piece:
                break
            chunk.extend(piece)
    text = chunk.decode("ascii", errors="ignore")
    result = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0] == "element" and parts[1] == "vertex":
            try:
                result["vertex_count"] = int(parts[2])
            except ValueError:
                pass
        if len(parts) >= 3 and parts[0] == "property" and parts[1] == "list":
            result.setdefault("list_properties", []).append(parts[3] if len(parts) > 3 else "?")
    return result


def _import_asset_from_path(
    repository: TourRepository,
    path: Path,
    *,
    source_kind: str,
    producer: str,
    provenance: dict | None = None,
    config: dict | None = None,
):
    data = path.read_bytes()
    points, _, colors_missing = parse_ply_data(data, config or {}, path.name)
    record = repository.assets.import_asset(
        str(path),
        role="scene",
        representation="point_cloud",
        source_kind=source_kind,
        producer=producer,
        coordinate_frame="asset",
        unit="arbitrary",
        provenance={
            "schema_version": 1,
            "colors_missing": colors_missing,
            **(provenance or {}),
        },
    )
    return record, points


def _create_scene(
    repository: TourRepository,
    record,
    points: np.ndarray,
    title: str,
    source_kind: str,
    config: dict,
    scene_id: str | None = None,
    entities=None,
    views=None,
) -> TourScene:
    transform = compute_display_transform(points)
    scene = TourScene(
        scene_id=scene_id or uuid.uuid4().hex,
        revision=1,
        title=title,
        asset_id=record.asset_id,
        asset_sha256=record.sha256,
        source_kind=source_kind,
        display_transform=transform,
        entities=entities or [],
        views=views or [],
        created_at=now(),
        updated_at=now(),
    )
    repository.ensure_preview(record, config)
    repository.create_scene(scene)
    return scene


def import_from_upload(repository: TourRepository, data: bytes, title: str, config: dict) -> TourScene:
    max_bytes = int(config["tour"]["max_upload_bytes"])
    if len(data) > max_bytes:
        raise PayloadTooLargeError(
            f"upload is {len(data)} bytes; limit is {max_bytes}"
        )
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as stream:
        stream.write(data)
        temp_path = Path(stream.name)
    try:
        record, points = _import_asset_from_path(
            repository,
            temp_path,
            source_kind="real",
            producer="tour_upload_v1",
            config=config,
        )
        return _create_scene(repository, record, points, title, "real", config)
    finally:
        try:
            temp_path.unlink()
        except OSError:
            pass


def import_from_reconstruction(
    repository: TourRepository,
    subject: str,
    output_dir,
    title: str | None,
    config: dict,
) -> TourScene:
    """Copy a validated reconstruction result into the tour assets.

    The path is resolved by the backend (subject name sanitized, confined to
    the configured reconstruction output directory); browser-supplied paths
    are never accepted.
    """
    clean = sanitize_name(subject)
    if clean != subject:
        raise InvalidDataError("invalid subject name")
    output_root = Path(output_dir).resolve()
    ply = (output_root / subject / "pointcloud.ply").resolve()
    if not ply.is_relative_to(output_root):
        raise InvalidDataError("subject escapes the reconstruction output directory")
    if not ply.is_file():
        raise NotFoundError(f"no point cloud for reconstruction subject {subject}")
    record, points = _import_asset_from_path(
        repository,
        ply,
        source_kind="real",
        producer="reconstruction_result_v1",
        provenance={"subject": subject},
        config=config,
    )
    return _create_scene(
        repository, record, points, title or f"重建结果 {subject}", "real", config
    )


def import_phase1(
    repository: TourRepository,
    phase1_root,
    title: str | None,
    config: dict,
) -> TourScene:
    """Import a Phase 1 composed scene's combined.ply as an immutable snapshot."""
    from src.scene.storage import SceneRepository

    root = Path(phase1_root).resolve()
    phase1_scene = SceneRepository(root).load("scene.json")
    combined = (root / "combined.ply").resolve()
    if not combined.is_relative_to(root) or not combined.is_file():
        raise NotFoundError("phase1 combined.ply not found")
    record, points = _import_asset_from_path(
        repository,
        combined,
        source_kind="synthetic",
        producer="phase1_combined_v1",
        provenance={"phase1_scene_id": phase1_scene.scene_id},
        config=config,
    )
    return _create_scene(
        repository, record, points, title or f"Phase 1 {phase1_scene.scene_id}", "synthetic", config
    )


# ---------------------------------------------------------------- synthetic demo
def seed_demo(repository: TourRepository, config: dict) -> dict:
    """Deterministic synthetic courtyard: three colored regions + ground.

    Explicitly ``source_kind="synthetic"``; never presented as reconstruction
    output. Creates entities, four views and a manual 3-shot fixture
    storyboard so the full offline loop is playable without an LLM.
    """
    from src.pointcloud_io import write_ply

    scene_id = "synthetic_courtyard"
    if (repository.root / "scenes" / scene_id).exists():
        raise ConflictError(f"demo scene {scene_id} already exists; choose a fresh output root")

    rng = np.random.default_rng(7)
    ground = np.array(
        [[x, 0.0, z] for x in np.linspace(-5, 5, 22) for z in np.linspace(-5, 5, 22)]
    )
    building = _box_points(rng, (-3.0, 0.0, -2.0), (2.0, 2.5, 2.0), 1400)
    tree = _box_points(rng, (1.8, 0.5, 0.8), (1.4, 2.0, 1.4), 1200)
    wall = _box_points(rng, (0.0, 0.0, -3.0), (3.0, 1.2, 1.0), 1000)
    colors = np.concatenate(
        [
            np.tile([150, 150, 155], (len(ground), 1)),
            np.tile([210, 70, 50], (len(building), 1)),
            np.tile([60, 165, 85], (len(tree), 1)),
            np.tile([60, 110, 220], (len(wall), 1)),
        ]
    ).astype(np.uint8)
    points = np.concatenate([ground, building, tree, wall])
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as stream:
        write_ply(points, colors, stream.name)
        temp_path = Path(stream.name)
    try:
        record, _ = _import_asset_from_path(
            repository,
            temp_path,
            source_kind="synthetic",
            producer="tour_seed_demo_v1",
            provenance={"fixture": "deterministic_courtyard_v1"},
            config=config,
        )
    finally:
        try:
            temp_path.unlink()
        except OSError:
            pass

    confirmed_at = now()
    entities = [
        Entity(
            entity_id="entity_building",
            label="红色建筑",
            description="庭院西侧的红色立方体区域（人工标注，非分割结果）",
            bounds_asset=[[-3.2, -0.1, -2.2], [-0.8, 2.6, 0.2]],
            facts=[
                Fact(fact_id="fact_building_region", text="建筑区域呈现为红色，轮廓近似立方体。", kind="descriptive"),
                Fact(fact_id="fact_building_west", text="建筑位于庭院的西侧（以全景视角为准）。", kind="spatial", scope_view_id="view_overview"),
            ],
            status="confirmed",
        ),
        Entity(
            entity_id="entity_tree",
            label="绿色树丛",
            description="庭院东侧的绿色区域（人工标注，非分割结果）",
            bounds_asset=[[1.7, 0.4, 0.7], [3.3, 2.6, 2.3]],
            facts=[
                Fact(fact_id="fact_tree_region", text="树丛区域为绿色，顶部较宽。", kind="descriptive"),
            ],
            status="confirmed",
        ),
        Entity(
            entity_id="entity_wall",
            label="蓝色矮墙",
            description="庭院南侧的蓝色矮墙区域（人工标注，非分割结果）",
            bounds_asset=[[-0.2, -0.1, -3.2], [3.2, 1.3, -1.8]],
            facts=[
                Fact(fact_id="fact_wall_region", text="矮墙为蓝色，高度低于建筑。", kind="descriptive"),
            ],
            status="confirmed",
        ),
    ]
    views = [
        ViewPreset(
            view_id="view_overview",
            name="全景",
            camera=Camera(position_asset=[0.0, 4.5, 7.0], target_asset=[0.0, 1.2, 0.0], up=[0.0, 1.0, 0.0], fov_y_deg=60.0, near_asset=0.01, far_asset=100.0, aspect_ratio=16.0 / 9.0),
            approved_entity_ids=["entity_building", "entity_tree", "entity_wall"],
            confirmation_source="synthetic_fixture",
            confirmed_at=confirmed_at,
        ),
        ViewPreset(
            view_id="view_building",
            name="建筑近景",
            camera=Camera(position_asset=[-2.0, 2.2, 3.5], target_asset=[-2.0, 1.2, 0.0], up=[0.0, 1.0, 0.0], fov_y_deg=55.0, near_asset=0.01, far_asset=100.0, aspect_ratio=16.0 / 9.0),
            approved_entity_ids=["entity_building"],
            confirmation_source="synthetic_fixture",
            confirmed_at=confirmed_at,
        ),
        ViewPreset(
            view_id="view_tree",
            name="树丛近景",
            camera=Camera(position_asset=[2.5, 2.0, 3.5], target_asset=[2.5, 1.2, 1.5], up=[0.0, 1.0, 0.0], fov_y_deg=55.0, near_asset=0.01, far_asset=100.0, aspect_ratio=16.0 / 9.0),
            approved_entity_ids=["entity_tree"],
            confirmation_source="synthetic_fixture",
            confirmed_at=confirmed_at,
        ),
        ViewPreset(
            view_id="view_wall",
            name="矮墙近景",
            camera=Camera(position_asset=[1.5, 1.5, -5.0], target_asset=[1.5, 0.6, -2.5], up=[0.0, 1.0, 0.0], fov_y_deg=55.0, near_asset=0.01, far_asset=100.0, aspect_ratio=16.0 / 9.0),
            approved_entity_ids=["entity_wall"],
            confirmation_source="synthetic_fixture",
            confirmed_at=confirmed_at,
        ),
    ]
    scene = _create_scene(
        repository,
        record,
        points,
        "合成庭院演示",
        "synthetic",
        config,
        scene_id=scene_id,
        entities=entities,
        views=views,
    )

    storyboard = Storyboard(
        storyboard_id="sb_synthetic_courtyard_01",
        revision=1,
        scene_id=scene.scene_id,
        scene_revision=scene.revision,
        asset_sha256=scene.asset_sha256,
        title="庭院导览",
        language="zh-CN",
        source="fixture",
        request_summary={"seed": "tour_seed_demo_v1"},
        shots=[
            Shot(shot_id="shot_01", view_id="view_overview",
                 focus_entity_ids=["entity_building", "entity_tree", "entity_wall"],
                 narration="我们先从整体俯瞰庭院：红色的建筑、绿色的树丛和蓝色的矮墙分布在不同位置。",
                 fact_ids=["fact_building_region", "fact_building_west", "fact_tree_region", "fact_wall_region"],
                 duration_seconds=10),
            Shot(shot_id="shot_02", view_id="view_building",
                 focus_entity_ids=["entity_building"],
                 narration="接下来走近这座红色建筑，可以看到它方正的体量与鲜明的颜色。",
                 fact_ids=["fact_building_region"],
                 duration_seconds=10),
            Shot(shot_id="shot_03", view_id="view_tree",
                 focus_entity_ids=["entity_tree"],
                 narration="最后把目光移到树丛，绿色区域在庭院东侧清晰可见。",
                 fact_ids=["fact_tree_region"],
                 duration_seconds=10),
        ],
        validation=StoryboardValidation(status="valid"),
        created_at=now(),
    )
    repository.create_storyboard(storyboard)
    return {
        "scene_id": scene.scene_id,
        "storyboard_id": storyboard.storyboard_id,
        "asset_sha256": scene.asset_sha256,
        "source_kind": "synthetic",
    }


def _box_points(rng, origin, size, count):
    low = np.asarray(origin)
    high = np.asarray(origin) + np.asarray(size)
    return rng.uniform(low, high, size=(count, 3))
