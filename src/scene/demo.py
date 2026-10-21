"""Small, deterministic synthetic fixture. No model or subprocess imports."""
from dataclasses import asdict
from itertools import product
import uuid
import numpy as np

from src.pointcloud_io import write_ply
from .composition import export, place
from .models import Anchor, Instance, PlacementConstraint, RunRecord, SceneDocument, Y_UP, now
from .storage import AssetRepository, SceneRepository, atomic_json, controlled
from .validation import validate_export


def demo(root, relation="left_of", right=(1., 0., 0.)):
    assets = AssetRepository(root)
    # Never overwrite an earlier demo.
    if controlled(assets.root, "scene.json").exists():
        raise FileExistsError("demo output already contains scene.json; choose a fresh --root")
    ground = np.array([[x, 0, z] for x in np.linspace(-5, 5, 11) for z in np.linspace(-5, 5, 11)])
    cube = np.array(list(product((-1., 1.), (0., 2.), (-1., 1.))))
    subject = np.array(list(product((2., 3.), (1., 4.), (-2., -1.))))
    records = []
    for name, p, color, role in (
        ("background", np.concatenate([ground, cube]), [90, 110, 130], "scene"),
        ("subject", subject, [220, 60, 30], "subject"),
    ):
        path = controlled(assets.root, f"fixture/{name}.ply")
        write_ply(p, np.tile(color, (len(p), 1)), path)
        records.append(assets.import_asset(
            path, role=role, representation="point_cloud", source_kind="synthetic",
            producer="phase1_cpu_fixture_v1", coordinate_frame=Y_UP,
            provenance={"fixture": "deterministic_cuboids", "schema_version": 1},
        ))
    scene = SceneDocument(
        "synthetic_demo",
        [Instance("background", records[0].asset_id, records[0].sha256),
         Instance("subject", records[1].asset_id, records[1].sha256)],
        [Anchor("box", [[-1, 0, -1], [1, 2, 1]], source="synthetic", instance_id="background")],
        reference_frame={"right": list(right)},
    )
    c = PlacementConstraint("subject", "box", relation, gap=.5, target_height=1.5)
    scene, _ = place(scene, c, assets)
    SceneRepository(assets).save(scene)
    artifact = export(scene, assets)
    result = validate_export(assets)
    atomic_json(controlled(assets.root, "validation.json"), result)
    run = RunRecord(uuid.uuid4().hex, "cpu_compose", "synthetic", {"constraint": asdict(c)},
                    status="succeeded" if result["passed"] else "failed",
                    started_at=now(), finished_at=now(), artifacts=[asdict(artifact)])
    atomic_json(controlled(assets.root, "run.json"), asdict(run))
    return result
