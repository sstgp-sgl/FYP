"""Headless point splats with per-pixel Z buffer; no invented surfaces or model inference."""
import numpy as np
from .geometry import project


def render(points, colors, instance_ids, shot, point_size=3, background=(239, 242, 247)):
    p, c, ids = np.asarray(points, float), np.asarray(colors), np.asarray(instance_ids)
    if not len(p):
        raise ValueError('empty point cloud')
    if c.shape != p.shape or c.dtype != np.uint8 or ids.shape != (len(p),) or ids.dtype.kind not in 'iu' or np.any(ids <= 0):
        raise ValueError('invalid colors/instance IDs')
    if type(point_size) is not int or not 1 <= point_size <= 15 or point_size % 2 != 1:
        raise ValueError('point_size must be odd, between 1 and 15')
    if len(background) != 3 or any(type(v) is not int or not 0 <= v <= 255 for v in background):
        raise ValueError('invalid background RGB')
    uv, depth, front = project(p, shot)
    h, w = shot['height'], shot['width']
    rgb = np.full((h, w, 3), background, dtype=np.uint8)
    zbuffer = np.full((h, w), np.inf)
    idbuffer = np.zeros((h, w), dtype=np.int32)
    radius = point_size // 2
    # Stable order: for equal depth the earlier point wins; reproducible tie policy.
    for i in np.flatnonzero(front):
        x, y = np.floor(uv[i] + .5).astype(int)
        if x+radius < 0 or x-radius >= w or y+radius < 0 or y-radius >= h:
            continue
        ys, xs = slice(max(0, y-radius), min(h, y+radius+1)), slice(max(0, x-radius), min(w, x+radius+1))
        mask = depth[i] < zbuffer[ys, xs]
        zbuffer[ys, xs][mask] = depth[i]
        idbuffer[ys, xs][mask] = ids[i]
        rgb[ys, xs][mask] = c[i]
    return rgb, idbuffer, zbuffer
