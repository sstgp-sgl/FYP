"""Independent acceptance over saved PLY points; does not import the solver."""
from dataclasses import asdict
from itertools import product
import numpy as np

from .models import ArtifactRef, version, number
from .storage import SceneRepository, controlled, read_json, read_vertices, sha256

# Fixed synthetic fixture scale, not estimated from a failed output.
FIXTURE_SCALE = 10.0
EPSILON = 1e-5 * max(1., FIXTURE_SCALE)


def validate_export(assets, scene_relative="scene.json", ply_relative="combined.ply"):
    scene = SceneRepository(assets).load(scene_relative)
    path = controlled(assets.root, ply_relative)
    index = read_json(path.with_suffix(".ranges.json"))
    version(index["schema_version"])
    artifact = ArtifactRef(**index["artifact"])
    if artifact.relative_path != ply_relative or artifact.sha256 != sha256(path) or index["scene_id"] != scene.scene_id:
        raise ValueError("export identity/hash mismatch")
    points, colors = read_vertices(path, "point_cloud")
    ranges = index["ranges"]
    expected = sorted([i for i in scene.instances if i.visible], key=lambda i: i.instance_id)
    if [r["instance_id"] for r in ranges] != [i.instance_id for i in expected]:
        raise ValueError("export instance order mismatch")
    chunks, checks = {}, []
    end = 0
    for r, instance in zip(ranges, expected):
        start, stop = r["start"], r["end"]
        if type(start) is not int or type(stop) is not int or start != end or not start < stop <= len(points):
            raise ValueError("invalid point range")
        end = stop
        p, c = assets.points(assets.get(instance.asset_id))
        q = points[start:stop]
        if len(q) != len(p):
            raise ValueError("point count mismatch")
        # Independently expand quaternion rather than call solver/transformed.
        x, y, z, w = instance.transform.rotation_xyzw
        xyz = np.array([x, y, z])
        rotation_result = p + 2*np.cross(xyz, np.cross(xyz, p)+w*p)
        intended = rotation_result*instance.transform.uniform_scale + instance.transform.translation_xyz
        matches = bool(np.allclose(q, intended, atol=EPSILON, rtol=0))
        color_match = bool(np.array_equal(c, colors[start:stop]))
        checks.append({"instance_id": instance.instance_id, "coordinates_match": matches, "colors_exact": color_match})
        chunks[instance.instance_id] = q
    if end != len(points):
        raise ValueError("unaccounted exported points")
    right = np.array(scene.reference_frame["right"])
    forward = np.cross(right, [0., 1., 0.])
    constraint_checks = []
    for c in scene.constraints:
        if c.subject_instance_id not in chunks:
            raise ValueError("constrained instance missing from export")
        q = chunks[c.subject_instance_id]
        a = next(a for a in scene.anchors if a.anchor_id == c.anchor_id)
        ac = np.array(list(product(*zip(*a.bounds))))
        ap, qp = ac@right, q@right
        gap = ap.min()-qp.max() if c.relation == "left_of" else qp.min()-ap.max()
        height, bottom = np.ptp(q[:, 1]), q[:, 1].min()
        center = (q.min(0)+q.max(0))/2
        forward_error = float((center-ac.mean(0))@forward)
        ok = abs(height-c.target_height) <= EPSILON and abs(bottom) <= EPSILON and abs(gap-c.gap) <= EPSILON and abs(forward_error) <= EPSILON
        constraint_checks.append({
            "subject_instance_id": c.subject_instance_id, "height": float(height),
            "bottom_y": float(bottom), "gap": float(gap), "forward_error": forward_error, "passed": bool(ok),
        })
    return {
        "schema_version": 1, "scene_id": scene.scene_id,
        "source": "synthetic" if all(assets.get(i.asset_id).source_kind == "synthetic" for i in expected) else "real",
        "epsilon": EPSILON, "fixture_scale": FIXTURE_SCALE,
        "scope": "exported point geometry and color only; no rendering/visual-quality claim",
        "instances": checks, "constraints": constraint_checks,
        "passed": all(c["coordinates_match"] and c["colors_exact"] for c in checks) and all(c["passed"] for c in constraint_checks),
    }
