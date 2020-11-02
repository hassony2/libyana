import torch
import numpy as np
from libyana.verify import checkshape


def batch_weakcam2persptrans(weak_cams, K, focal_scale):
    checkshape.check_shape(K, (-1, 3, 3), "K")
    checkshape.check_shape(weak_cams, (-1, 3), "weak_cams")
    tz = focal_scale * (K[:, 1, 1] + K[:, 0, 0]) / weak_cams[:, 0] / 2
    tx = (weak_cams[:, 1] - K[:, 0, 2]) * tz / K[:, 0, 0]
    ty = (weak_cams[:, 2] - K[:, 1, 2]) * tz / K[:, 1, 1]
    return torch.stack([tx, ty, tz], -1)


def weakcam2persptrans(weak_cam, K, focal_scale=1):
    checkshape.check_shape(K, (3, 3), "K")
    checkshape.check_shape(weak_cam, (3,), "weak_cams")
    tz = focal_scale * (K[1, 1] + K[0, 0]) / weak_cam[0] / 2
    tx = (weak_cam[1] - K[0, 2]) * tz / K[0, 0]
    ty = (weak_cam[2] - K[1, 2]) * tz / K[1, 1]
    return np.array([tx, ty, tz])
