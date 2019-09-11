import numpy as np
import torch

from libyana.meshutils import meshio
from libyana.meshutils import vertexsamplept
from libyana.meshutils import vertexsample


def test_points_from_mesh():
    info = meshio.faster_load_obj("test/assets/model_normalized.obj")
    verts = info["vertices"]
    faces = info["faces"]
    verts_pt = torch.Tensor(verts)
    faces_pt = torch.Tensor(faces).long()
    vertex_nb = 600

    tri_area_np = vertexsample.tri_area(verts[faces])
    tri_area_pt = vertexsamplept.tri_area(verts_pt[faces_pt])
    assert np.allclose(tri_area_pt.numpy(), tri_area_np)
    sampled_points = vertexsamplept.points_from_mesh(
        faces_pt, verts_pt, vertex_nb=vertex_nb, show_cloud=True
    )
    sampled_points.numpy().shape == (vertex_nb, 3)
