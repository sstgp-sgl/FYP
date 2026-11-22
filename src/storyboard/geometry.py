"""Y-up world, column vectors; right-handed OpenCV camera (+X right,+Y down,+Z forward)."""
import numpy as np


def camera_from_look_at(eye, target, width, height, fov_degrees, near, far):
    eye, target = np.asarray(eye, dtype=float), np.asarray(target, dtype=float)
    forward = target - eye
    if eye.shape != (3,) or not np.isfinite([eye, target]).all() or np.linalg.norm(forward) < 1e-12:
        raise ValueError('invalid camera eye/target')
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 1., 0.])
    if np.linalg.norm(right) < 1e-12:
        raise ValueError('camera parallel to world up')
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    transform = np.eye(4)
    transform[:3, :3] = np.column_stack([right, down, forward])
    transform[:3, 3] = eye
    focal = height / (2 * np.tan(np.deg2rad(fov_degrees) / 2))
    shot = dict(schema_version=1, projection='perspective', convention='opencv_x_right_y_down_z_forward',
                width=width, height=height, near=near, far=far,
                K=[[focal, 0, (width-1)/2], [0, focal, (height-1)/2], [0, 0, 1]],
                T_world_from_camera=transform.tolist())
    validate_camera(shot)
    return shot


def validate_camera(shot):
    if shot['schema_version'] != 1 or shot['projection'] != 'perspective' or shot['convention'] != 'opencv_x_right_y_down_z_forward':
        raise ValueError('unsupported camera convention')
    w, h = shot['width'], shot['height']
    if type(w) is not int or type(h) is not int or not 1 <= w <= 4096 or not 1 <= h <= 4096:
        raise ValueError('invalid render resolution')
    k, t = np.asarray(shot['K'], float), np.asarray(shot['T_world_from_camera'], float)
    if k.shape != (3, 3) or t.shape != (4, 4) or not np.isfinite(k).all() or not np.isfinite(t).all():
        raise ValueError('invalid camera matrices')
    if not np.allclose(k[2], [0, 0, 1]) or k[0, 0] <= 0 or k[1, 1] <= 0 or k[0, 1] != 0 or k[1, 0] != 0:
        raise ValueError('unsupported intrinsic matrix')
    if not np.allclose(t[3], [0, 0, 0, 1]) or not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-8) or not np.isclose(np.linalg.det(t[:3, :3]), 1):
        raise ValueError('camera must be a rigid right-handed transform')
    if not np.isfinite([shot['near'], shot['far']]).all() or not 0 < shot['near'] < shot['far']:
        raise ValueError('invalid near/far')
    return k, t


def project(points, shot):
    k, t = validate_camera(shot)
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise ValueError('invalid points')
    camera = (p - t[:3, 3]) @ t[:3, :3]
    depth = camera[:, 2]
    visible = (depth >= shot['near']) & (depth <= shot['far'])
    uv = np.full((len(p), 2), np.nan)
    uv[visible] = (camera[visible] @ k.T)[:, :2] / depth[visible, None]
    return uv, depth, visible
