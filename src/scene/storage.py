"""Controlled paths, atomic JSON documents and immutable copied assets."""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import uuid

import numpy as np
from .models import AssetRecord, SceneDocument, identifier, now


def controlled(root, relative):
    root = Path(root).resolve()
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts or not rel.parts:
        raise ValueError("unsafe relative path")
    result = (root / rel).resolve()
    if not result.is_relative_to(root) or result == root:
        raise ValueError("path escapes controlled root")
    return result


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    # Serialize before touching the old document; os.replace is same-filesystem.
    data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path):
    def invalid(value):
        raise ValueError("non-finite JSON number: " + value)
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def require_versioned(data, *keys):
    if not isinstance(data, dict) or "schema_version" not in data:
        raise ValueError("persistent object is missing schema_version")
    for key in keys:
        for child in data.get(key, []):
            if not isinstance(child, dict) or "schema_version" not in child:
                raise ValueError(f"persistent {key} entry is missing schema_version")
            if key == "instances" and ("transform" not in child or "schema_version" not in child["transform"]):
                raise ValueError("persistent transform is missing schema_version")
    return data


GAUSSIAN_FIELDS = (
    "x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
    "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
)


def read_vertices(path, representation):
    # plyfile handles both binary and ASCII; no GPU/model imports.
    from plyfile import PlyData
    ply = PlyData.read(str(path))
    v = ply["vertex"].data
    fields = set(v.dtype.names or ())
    required = {"x", "y", "z", "red", "green", "blue"} if representation == "point_cloud" else set(GAUSSIAN_FIELDS)
    if representation not in {"point_cloud", "gaussian_splat"} or not required <= fields or len(v) == 0:
        raise ValueError("invalid or empty " + representation + " PLY")
    if representation == "point_cloud" and (set(GAUSSIAN_FIELDS) - {"x", "y", "z"}) & fields:
        raise ValueError("Gaussian fields cannot be imported as a surface point cloud")
    # Reject NaN/Inf in all scalar numeric fields, including optional SH bands.
    for name in fields:
        if v[name].dtype.kind not in "fiu" or not np.isfinite(v[name]).all():
            raise ValueError("invalid numeric PLY field: " + name)
    points = np.column_stack([v[k] for k in ("x", "y", "z")]).astype(np.float64)
    if representation == "point_cloud":
        for k in ("red", "green", "blue"):
            if v[k].dtype.kind != "u" or v[k].dtype.itemsize != 1:
                raise ValueError("RGB fields must be uchar")
        colors = np.column_stack([v[k] for k in ("red", "green", "blue")])
        return points, colors
    # Official save convention: log-scale, opacity logit, quaternion wxyz.
    # Negative scale/opacity values are valid stored values.
    rotations = np.column_stack([v[f"rot_{i}"] for i in range(4)]).astype(np.float64)
    norms = np.linalg.norm(rotations, axis=1)
    if np.any(np.abs(norms - 1) > 1e-3):
        raise ValueError("Gaussian rotation must be a unit quaternion")
    scales = np.column_stack([v[f"scale_{i}"] for i in range(3)]).astype(np.float64)
    with np.errstate(over="ignore", under="ignore"):
        actual = np.exp(scales)
    if not np.isfinite(actual).all() or np.any(actual <= 0):
        raise ValueError("invalid decoded Gaussian scale")
    return points, None


class AssetRepository:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def import_asset(self, path, *, role, representation, source_kind, producer,
                     coordinate_frame, unit="arbitrary", **metadata):
        asset_id = uuid.uuid4().hex
        directory = controlled(self.root, f"assets/{asset_id}")
        directory.mkdir(parents=True, exist_ok=False)
        target = directory / "data.ply"
        # Validate the copied bytes, not a potentially changed external file.
        try:
            shutil.copyfile(path, target)
            points, _ = read_vertices(target, representation)
            bounds = [points.min(0).tolist(), points.max(0).tolist()] if representation == "point_cloud" else None
            record = AssetRecord(
                asset_id=asset_id, role=role, representation=representation,
                relative_path=str(target.relative_to(self.root)), sha256=sha256(target),
                source_kind=source_kind, producer=producer, coordinate_frame=coordinate_frame,
                unit=unit, created_at=now(), bounds=bounds, point_count=len(points), **metadata,
            )
            atomic_json(directory / "asset.json", asdict(record))
            return record
        except BaseException:
            # This directory was just created for this import and has no previous data.
            shutil.rmtree(directory, ignore_errors=True)
            raise

    def get(self, asset_id):
        identifier(asset_id)
        data = read_json(controlled(self.root, f"assets/{asset_id}/asset.json"))
        require_versioned(data)
        record = AssetRecord(**data)
        if record.asset_id != asset_id:
            raise ValueError("asset identity mismatch")
        self.verify(record)
        return record

    def verify(self, record):
        record = AssetRecord(**asdict(record))
        path = controlled(self.root, record.relative_path)
        if not path.is_file():
            raise FileNotFoundError("missing asset: " + record.asset_id)
        if sha256(path) != record.sha256:
            raise ValueError("asset hash mismatch: " + record.asset_id)
        return path

    def points(self, record):
        if record.representation != "point_cloud":
            raise ValueError("unsupported_representation: gaussian_splat")
        return read_vertices(self.verify(record), "point_cloud")


class SceneRepository:
    def __init__(self, assets):
        self.assets = assets

    def validate(self, scene):
        scene = SceneDocument(**asdict(scene))
        for instance in scene.instances:
            asset = self.assets.get(instance.asset_id)
            if asset.sha256 != instance.asset_hash:
                raise ValueError("scene asset hash mismatch")
            if asset.coordinate_frame != scene.coordinate_frame or asset.unit != scene.unit:
                raise ValueError("asset requires explicit coordinate/unit alignment")
        return scene

    def save(self, scene, relative="scene.json"):
        scene = self.validate(scene)
        path = controlled(self.assets.root, relative)
        if "assets" in Path(relative).parts or "runs" in Path(relative).parts:
            raise ValueError("scene document cannot overwrite managed assets or run records")
        atomic_json(path, asdict(scene))

    def load(self, relative="scene.json"):
        data = require_versioned(read_json(controlled(self.assets.root, relative)), "instances", "anchors", "constraints")
        return self.validate(SceneDocument(**data))
