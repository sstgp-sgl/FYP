"""Version 1 contracts. Coordinates use column vectors and quaternion xyzw."""
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import math
import re

VERSION = 1
Y_UP = "right_handed_y_up"
UNITS = {"arbitrary", "estimated_metric", "calibrated_metric"}


def now():
    return datetime.now(timezone.utc).isoformat()


def version(value):
    if type(value) is not int or value != VERSION:
        raise ValueError("unsupported schema_version")


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", value):
        raise ValueError("invalid identifier")
    return value


def number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("expected finite number")
    return float(value)


def vector(value, length):
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError("wrong vector shape")
    return [number(x) for x in value]


def digest(value):
    if not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value):
        raise ValueError("invalid sha256")


def bbox(value):
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("bounds must be [minimum_xyz, maximum_xyz]")
    lo, hi = vector(value[0], 3), vector(value[1], 3)
    if any(a > b for a, b in zip(lo, hi)):
        raise ValueError("inverted bounds")
    return [lo, hi]


@dataclass
class Transform:
    translation_xyz: list = field(default_factory=lambda: [0., 0., 0.])
    rotation_xyzw: list = field(default_factory=lambda: [0., 0., 0., 1.])
    uniform_scale: float = 1.
    schema_version: int = VERSION

    def __post_init__(self):
        version(self.schema_version)
        self.translation_xyz = vector(self.translation_xyz, 3)
        q = vector(self.rotation_xyzw, 4)
        norm = math.hypot(*q)
        if norm == 0:
            raise ValueError("zero quaternion")
        q = [v / norm for v in q]
        # Unique sign, including rotations with w=0; stable save/load.
        pivot = next((x for x in reversed(q) if x != 0), 1)
        self.rotation_xyzw = [-x for x in q] if pivot < 0 else q
        self.uniform_scale = number(self.uniform_scale)
        if self.uniform_scale <= 0:
            raise ValueError("scale must be positive")

    def matrix(self):
        import numpy as np
        x, y, z, w = self.rotation_xyzw
        r = np.array([
            [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
        ])
        m = np.eye(4)
        m[:3, :3] = self.uniform_scale * r
        m[:3, 3] = self.translation_xyz
        return m


@dataclass
class AssetRecord:
    asset_id: str
    role: str
    representation: str
    relative_path: str
    sha256: str
    source_kind: str
    producer: str
    coordinate_frame: str
    unit: str
    created_at: str
    schema_version: int = VERSION
    source_image_hash: str | None = None
    camera_metadata: dict | None = None
    bounds: list | None = None
    source_to_canonical: dict | None = None
    point_count: int | None = None
    preprocessing: dict | None = None
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        version(self.schema_version)
        identifier(self.asset_id)
        digest(self.sha256)
        if self.source_image_hash is not None:
            digest(self.source_image_hash)
        if self.role not in {"scene", "subject"} or self.representation not in {"point_cloud", "gaussian_splat"}:
            raise ValueError("invalid asset type")
        if self.source_kind not in {"real", "synthetic"} or self.unit not in UNITS:
            raise ValueError("invalid provenance/unit")
        for value in (self.producer, self.coordinate_frame, self.created_at, self.relative_path):
            if not isinstance(value, str) or not value:
                raise ValueError("missing asset metadata")
        if self.point_count is not None and (type(self.point_count) is not int or self.point_count <= 0):
            raise ValueError("invalid point_count")
        if self.bounds is not None:
            self.bounds = bbox(self.bounds)
        if self.representation == "gaussian_splat" and self.bounds is not None:
            raise ValueError("Gaussian bounds are not surface bounds")
        if self.source_to_canonical is not None:
            self.source_to_canonical = asdict(Transform(**self.source_to_canonical))


@dataclass
class Instance:
    instance_id: str
    asset_id: str
    asset_hash: str
    transform: Transform = field(default_factory=Transform)
    visible: bool = True
    schema_version: int = VERSION

    def __post_init__(self):
        version(self.schema_version)
        identifier(self.instance_id)
        identifier(self.asset_id)
        digest(self.asset_hash)
        if isinstance(self.transform, dict):
            self.transform = Transform(**self.transform)
        if not isinstance(self.transform, Transform) or type(self.visible) is not bool:
            raise ValueError("invalid instance")


@dataclass
class Anchor:
    anchor_id: str
    bounds: list
    source: str = "manual"
    instance_id: str | None = None
    schema_version: int = VERSION

    def __post_init__(self):
        version(self.schema_version)
        identifier(self.anchor_id)
        self.bounds = bbox(self.bounds)
        if self.source not in {"manual", "synthetic"}:
            raise ValueError("invalid anchor source")
        if self.instance_id is not None:
            identifier(self.instance_id)


@dataclass
class PlacementConstraint:
    subject_instance_id: str
    anchor_id: str
    relation: str
    gap: float
    target_height: float
    frame: str = "reference_ground"
    support: str = "ground"
    schema_version: int = VERSION

    def __post_init__(self):
        version(self.schema_version)
        identifier(self.subject_instance_id)
        identifier(self.anchor_id)
        self.gap, self.target_height = number(self.gap), number(self.target_height)
        if self.relation not in {"left_of", "right_of"} or self.frame != "reference_ground" or self.support != "ground":
            raise ValueError("unsupported constraint")
        if self.gap < 0 or self.target_height <= 0:
            raise ValueError("invalid gap/height")


@dataclass
class SceneDocument:
    scene_id: str
    instances: list
    anchors: list
    constraints: list = field(default_factory=list)
    coordinate_frame: str = Y_UP
    unit: str = "arbitrary"
    ground: dict = field(default_factory=lambda: {"y": 0.})
    reference_frame: dict = field(default_factory=lambda: {"right": [1., 0., 0.]})
    updated_at: str = field(default_factory=now)
    schema_version: int = VERSION

    def __post_init__(self):
        version(self.schema_version)
        identifier(self.scene_id)
        if self.coordinate_frame != Y_UP or self.unit not in UNITS:
            raise ValueError("unsupported scene coordinates/unit")
        if set(self.ground) != {"y"} or number(self.ground["y"]) != 0:
            raise ValueError("only ground y=0 supported")
        if set(self.reference_frame) != {"right"}:
            raise ValueError("invalid reference frame")
        right = vector(self.reference_frame["right"], 3)
        norm = math.hypot(*right)
        if norm == 0 or abs(right[1]) > 1e-12:
            raise ValueError("reference right must be nonzero and horizontal")
        self.reference_frame = {"right": [right[0]/norm, 0., right[2]/norm]}
        for key, cls in (("instances", Instance), ("anchors", Anchor), ("constraints", PlacementConstraint)):
            values = getattr(self, key)
            setattr(self, key, [cls(**v) if isinstance(v, dict) else cls(**asdict(v)) for v in values])
        ids = [i.instance_id for i in self.instances]
        anchors = [a.anchor_id for a in self.anchors]
        if len(ids) != len(set(ids)) or len(anchors) != len(set(anchors)):
            raise ValueError("duplicate scene ID")
        for a in self.anchors:
            if a.instance_id is not None and a.instance_id not in ids:
                raise ValueError("unknown anchor instance")
        seen = set()
        for c in self.constraints:
            if c.subject_instance_id not in ids or c.anchor_id not in anchors:
                raise ValueError("unknown constraint ID")
            if c.subject_instance_id in seen:
                raise ValueError("only one placement constraint per instance")
            seen.add(c.subject_instance_id)


@dataclass
class ArtifactRef:
    relative_path: str
    sha256: str
    representation: str
    schema_version: int = VERSION

    def __post_init__(self):
        version(self.schema_version)
        digest(self.sha256)
        if self.representation not in {"point_cloud", "gaussian_splat", "json"}:
            raise ValueError("invalid artifact representation")


@dataclass
class RunRecord:
    run_id: str
    backend: str
    source: str
    parameters: dict
    status: str = "pending"
    started_at: str | None = None
    finished_at: str | None = None
    duration_seconds: float | None = None
    exit_code: int | None = None
    logs: list = field(default_factory=list)
    artifacts: list = field(default_factory=list)
    error: dict | None = None
    gpu_peak_memory_bytes: int | None = None
    schema_version: int = VERSION

    def __post_init__(self):
        version(self.schema_version)
        identifier(self.run_id)
        if self.status not in {"pending", "running", "succeeded", "failed", "interrupted"}:
            raise ValueError("invalid run status")
        if self.source not in {"real", "synthetic", "test"}:
            raise ValueError("invalid run source")
        if self.duration_seconds is not None and number(self.duration_seconds) < 0:
            raise ValueError("invalid run duration")
        self.artifacts = [asdict(ArtifactRef(**a)) for a in self.artifacts]
