"""Real PLY registration, explicit orientation proposals, and one-time canonical adaptation.

Raw registered PLY bytes/RGB never change. SceneRepository sees canonical points through
an adapter; subsequent Instance transforms are applied exactly once by the old exporter.
"""
from dataclasses import replace, asdict
from pathlib import Path
import shutil
import numpy as np
from src.scene.models import Transform, Y_UP
from src.scene.storage import AssetRepository, atomic_json, read_json, sha256, controlled
from src.scene.composition import transformed


def quaternion(rotation):
    r = np.asarray(rotation, float)
    # Symmetric eigenproblem, quaternion xyzw. Sign canonicalized by Transform.
    k = np.array([[r[0,0]-r[1,1]-r[2,2], r[0,1]+r[1,0], r[0,2]+r[2,0], r[2,1]-r[1,2]],
                  [r[0,1]+r[1,0], r[1,1]-r[0,0]-r[2,2], r[1,2]+r[2,1], r[0,2]-r[2,0]],
                  [r[0,2]+r[2,0], r[1,2]+r[2,1], r[2,2]-r[0,0]-r[1,1], r[1,0]-r[0,1]],
                  [r[2,1]-r[1,2], r[0,2]-r[2,0], r[1,0]-r[0,1], np.trace(r)]]) / 3
    _, vec = np.linalg.eigh(k)
    return vec[:, -1].tolist()


def canonical_transform(points, source_up, source_forward, target_height, yaw_degrees=0):
    p = np.asarray(points, float)
    up, front = np.asarray(source_up, float), np.asarray(source_forward, float)
    if up.shape != (3,) or front.shape != (3,) or not np.isfinite([up, front]).all() or np.linalg.norm(up) < 1e-10:
        raise ValueError('invalid source up/forward')
    up = up / np.linalg.norm(up)
    front = front - up * (front @ up)
    if np.linalg.norm(front) < 1e-10:
        raise ValueError('source forward parallel to up')
    front /= np.linalg.norm(front)
    if not np.isfinite([target_height, yaw_degrees]).all() or target_height <= 0:
        raise ValueError('invalid target height/yaw')
    r = np.vstack([np.cross(up, front), up, front])
    a = np.deg2rad(yaw_degrees)
    r = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]) @ r
    q = p @ r.T
    lo, hi = q.min(0), q.max(0)
    height = hi[1] - lo[1]
    if height <= 1e-10:
        raise ValueError('zero canonical height')
    s = target_height / height
    bottom = np.array([(lo[0]+hi[0])/2, lo[1], (lo[2]+hi[2])/2])
    return Transform(translation_xyz=(-bottom*s).tolist(), rotation_xyzw=quaternion(r), uniform_scale=float(s))


def propose_camera_axes(camera_path):
    c = read_json(camera_path)['cameras'][0]
    r = np.asarray(c['extrinsic'], float)[:3, :3]
    if r.shape != (3, 3) or not np.isfinite(r).all():
        raise ValueError('invalid source camera')
    return dict(source_up=(-r[1]).tolist(), source_forward=(-r[2]).tolist(),
                orientation_basis='proposal_from_camera_0_not_user_confirmed')


def import_asset(library, ply, metadata):
    library = Path(library).resolve()
    m = dict(metadata)
    if not m.get('display_name') or not m.get('producer') or not m.get('source_image'):
        raise ValueError('display_name, producer and source_image are required')
    if m.get('source_kind', 'real') != 'real':
        raise ValueError('real asset importer requires source_kind=real')
    source = Path(m['source_image'])
    if not source.is_file():
        raise FileNotFoundError('missing source image')
    has_axes = m.get('source_up') is not None and m.get('source_forward') is not None
    if m.get('confirmed_by_user', False) and (not has_axes or m.get('confirmation_actor') != 'user'):
        raise ValueError('confirmation requires explicit axes and user confirmation_actor')
    from PIL import Image
    with Image.open(source) as im: im.verify()
    repo = AssetRepository(library)
    # Reuse strict ASCII/binary PLY field, finite-value, RGB, Gaussian and empty validation.
    from src.scene.storage import read_vertices
    points, _ = read_vertices(ply, 'point_cloud')
    if np.max(np.ptp(points, axis=0)) <= 1e-10:
        raise ValueError('degenerate 3D bounds')
    transform = canonical_transform(points, m['source_up'], m['source_forward'], m.get('target_height', 1.5), m.get('yaw_degrees', 0)) if has_axes else None
    rec = repo.import_asset(ply, role='subject', representation='point_cloud', source_kind='real',
        producer=m['producer'], coordinate_frame='source_reconstruction_world', source_image_hash=sha256(source),
        source_to_canonical=asdict(transform) if transform else None,
        provenance=dict(source_ply_sha256=sha256(ply), source_ply=str(Path(ply).resolve()),
                        reconstruction=m.get('reconstruction', {}), raw_prediction_reference=m.get('raw_prediction_reference'),
                        filtering='source_result_preserved_no_additional_color_filter'))
    folder = library / 'assets' / rec.asset_id
    image_name = 'source_image' + source.suffix.lower()
    shutil.copyfile(source, folder / image_name)
    from PIL import Image
    with Image.open(folder/image_name) as im: im.verify()
    canonical = transformed(points, transform) if transform else None
    side = dict(schema_version=1, asset_id=rec.asset_id, display_name=m['display_name'],
        display_name_confirmed=bool(m.get('display_name_confirmed', False)),
        confirmed_by_user=bool(m.get('confirmed_by_user', False)),
        orientation_status=('confirmed' if m.get('confirmed_by_user') else 'proposed') if transform else 'needs_orientation',
        orientation_basis=m.get('orientation_basis', 'explicit_metadata_axes' if has_axes else 'unknown'),
        source_image=image_name, source_image_sha256=sha256(folder/image_name),
        source_up=m.get('source_up'), source_forward=m.get('source_forward'), yaw_degrees=m.get('yaw_degrees', 0),
        target_height=m.get('target_height', 1.5), source_to_canonical=rec.source_to_canonical,
        canonical_bounds=[canonical.min(0).tolist(), canonical.max(0).tolist()] if canonical is not None else None,
        quality_flags=m.get('quality_flags', []), metadata_input=m)
    atomic_json(folder/'semantics.json', side)
    return side


class CanonicalAssetRepository(AssetRepository):
    """Read raw bytes; canonical transform is applied once only, before Instance transform."""
    def __init__(self, root):
        super().__init__(root)
        self._cache = {}

    def _canonical(self, raw):
        key = (raw.asset_id, raw.sha256, repr(raw.source_to_canonical))
        if key not in self._cache:
            if raw.source_to_canonical is None:
                raise ValueError('needs_orientation: ' + raw.asset_id)
            p, c = AssetRepository.points(self, raw)
            self._cache[key] = (transformed(p, Transform(**raw.source_to_canonical)), c)
        return self._cache[key]

    def semantics(self, asset_id):
        return read_json(controlled(self.root, f'assets/{asset_id}/semantics.json'))

    def get(self, asset_id):
        raw = super().get(asset_id)
        if raw.source_to_canonical is None:
            raise ValueError('needs_orientation: ' + asset_id)
        q, _ = self._canonical(raw)
        return replace(raw, coordinate_frame=Y_UP, bounds=[q.min(0).tolist(), q.max(0).tolist()])

    def points(self, record):
        raw = AssetRepository.get(self, record.asset_id)
        return self._canonical(raw)


def freeze_assets(library, asset_ids, run_root):
    src, dst = Path(library).resolve(), Path(run_root).resolve()
    repo = AssetRepository(src)
    for asset_id in dict.fromkeys(asset_ids):
        rec = repo.get(asset_id)
        source = controlled(src, 'assets/' + asset_id)
        shutil.copytree(source, dst/'assets'/asset_id)
        frozen = AssetRepository(dst).get(asset_id)
        assert frozen.sha256 == rec.sha256
    return CanonicalAssetRepository(dst)
