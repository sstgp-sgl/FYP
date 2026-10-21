"""Deterministic CPU layout. Immutable source points are never rewritten."""
from dataclasses import asdict
from itertools import product
import numpy as np

from src.pointcloud_io import write_ply
from .models import ArtifactRef, PlacementConstraint, SceneDocument, Transform, Y_UP, now
from .storage import SceneRepository, atomic_json, controlled, sha256


def corners(bounds):
    return np.array(list(product(*zip(*bounds))), dtype=np.float64)


def transformed(points, transform):
    matrix = transform.matrix()
    result = points @ matrix[:3, :3].T + matrix[:3, 3]
    if not np.isfinite(result).all():
        raise ValueError("non-finite transformed points")
    return result


def place(scene, constraint, assets):
    scene = SceneRepository(assets).validate(scene)
    constraint = PlacementConstraint(**asdict(constraint))
    instance = next((i for i in scene.instances if i.instance_id == constraint.subject_instance_id), None)
    anchor = next((a for a in scene.anchors if a.anchor_id == constraint.anchor_id), None)
    if instance is None or anchor is None:
        raise ValueError("unknown constraint ID")
    if anchor.instance_id == instance.instance_id:
        raise ValueError("self anchor is unsupported")
    if not instance.visible:
        raise ValueError("cannot place invisible instance")
    asset = assets.get(instance.asset_id)
    if asset.role != "subject" or asset.coordinate_frame != Y_UP:
        raise ValueError("layout requires confirmed Y-up subject")
    p, _ = assets.points(asset)
    lo, hi = p.min(0), p.max(0)
    if hi[1] <= lo[1]:
        raise ValueError("zero-height subject")
    bottom = np.array([(lo[0]+hi[0])/2, lo[1], (lo[2]+hi[2])/2])
    scale = constraint.target_height / (hi[1]-lo[1])
    centered = (p-bottom)*scale
    right = np.array(scene.reference_frame["right"])
    forward = np.cross(right, [0., 1., 0.])
    ac = corners(anchor.bounds)
    projection = ac @ right
    subject_projection = centered @ right
    offset = (projection.min()-constraint.gap-subject_projection.max()
              if constraint.relation == "left_of"
              else projection.max()+constraint.gap-subject_projection.min())
    forward_offset = np.mean(ac, axis=0) @ forward
    translation = -scale*bottom + offset*right + forward_offset*forward
    instance.transform = Transform(translation_xyz=translation.tolist(), uniform_scale=float(scale))
    scene.constraints = [c for c in scene.constraints if c.subject_instance_id != instance.instance_id] + [constraint]
    scene.updated_at = now()
    # Derived feedback only. validate_export independently re-reads exported data.
    q = transformed(p, instance.transform)
    gap = (projection.min()-(q@right).max() if constraint.relation == "left_of"
           else (q@right).min()-projection.max())
    return scene, {"height": float(np.ptp(q[:, 1])), "bottom_y": float(q[:, 1].min()), "gap": float(gap)}


def export(scene, assets, relative="combined.ply"):
    scene = SceneRepository(assets).validate(scene)
    # Even hidden Gaussian instances are not silently treated as point clouds.
    if any(assets.get(i.asset_id).representation != "point_cloud" for i in scene.instances):
        raise ValueError("unsupported_representation: gaussian_splat")
    all_points, all_colors, ranges = [], [], []
    start = 0
    for instance in sorted(scene.instances, key=lambda i: i.instance_id):
        if not instance.visible:
            continue
        p, rgb = assets.points(assets.get(instance.asset_id))
        all_points.append(transformed(p, instance.transform))
        all_colors.append(rgb)
        ranges.append({"instance_id": instance.instance_id, "start": start, "end": start+len(p)})
        start += len(p)
    if not all_points:
        raise ValueError("empty export")
    path = controlled(assets.root, relative)
    if {"assets", "runs"} & set(path.relative_to(assets.root).parts):
        raise ValueError("export cannot overwrite managed assets or run records")
    write_ply(np.concatenate(all_points), np.concatenate(all_colors), path)
    artifact = ArtifactRef(relative_path=relative, sha256=sha256(path), representation="point_cloud")
    atomic_json(path.with_suffix(".ranges.json"), {
        "schema_version": 1, "scene_id": scene.scene_id,
        "artifact": asdict(artifact), "ranges": ranges,
    })
    return artifact
