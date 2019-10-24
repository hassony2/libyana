import numpy as np


def center_vert_bbox(vertices, bbox_center=None, bbox_scale=None, scale=True):
    if bbox_center is None:
        bbox_center = (vertices.min(0) + vertices.max(0)) / 2
    vertices = vertices - bbox_center
    if scale:
        if bbox_scale is None:
            bbox_scale = np.linalg.norm(vertices, 2, 1).max()
        vertices = vertices / bbox_scale
    else:
        bbox_scale = 1
    return vertices, bbox_center, bbox_scale


def center_vert_pt(vertices, scale=True):
    bbox_center = (vertices.min(0)[0] + vertices.max(0)[0]) / 2
    vertices = vertices - bbox_center
    if scale:
        max_norm = vertices.norm(2, 1).max()
        vertices = vertices / max_norm
    else:
        max_norm = 1
    return vertices, bbox_center, max_norm
