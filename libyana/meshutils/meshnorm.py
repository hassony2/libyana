import numpy as np


def center_vert_bbox(vertices, bbox_center=None, bbox_scale=None):
    if bbox_center is None:
        bbox_center = (vertices.min(0) + vertices.max(0)) / 2
    vertices = vertices - bbox_center
    if bbox_scale is None:
        bbox_scale = np.linalg.norm(vertices, 2, 1).max()
    vertices = vertices / bbox_scale
    return vertices, bbox_center, bbox_scale


def center_vert_pt(vertices):
    bbox_center = (vertices.min(0)[0] + vertices.max(0)[0]) / 2
    vertices = vertices - bbox_center
    max_norm = vertices.norm(2, 1).max()
    vertices = vertices / max_norm
    return vertices, bbox_center, max_norm
