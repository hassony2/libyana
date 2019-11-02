import torch
import torch.nn.functional as torch_f


def compute_normals(verts, faces):
    # adapted from https://github.com/rusty1s/pytorch_geometric
    vec1 = verts[faces[:, 1]] - verts[faces[:, 0]]
    vec2 = verts[faces[:, 2]] - verts[faces[:, 0]]
    face_norm = torch_f.normalize(vec1.cross(vec2), p=2, dim=-1)  # [F, 3]

    idx = torch.cat([faces[:, 0], faces[:, 1], faces[:, 2]], dim=0)
    face_norm = face_norm.repeat(3, 1)

    norm = torch.zeros_like(verts).scatter_add(
        0, idx.unsqueeze(1).repeat(1, 3), face_norm
    )
    norm = torch_f.normalize(norm, p=2, dim=-1)  # [N, 3]
    return norm, face_norm, idx


def batch_compute_normals(vertices, faces):
    p1 = torch.gather(vertices, 1, faces[:, :, 0:1].repeat(1, 1, 3))
    p2 = torch.gather(vertices, 1, faces[:, :, 1:2].repeat(1, 1, 3))
    p3 = torch.gather(vertices, 1, faces[:, :, 2:].repeat(1, 1, 3))

    v1 = p2 - p1
    v2 = p3 - p1
    face_norms = torch.cross(v1, v2)
    face_norms = torch_f.normalize(face_norms, 2, -1)
    flat_face_idxs = torch.cat(
        [faces[:, :, 0], faces[:, :, 1], faces[:, :, 2]], -1
    )
    flat_face_norms = face_norms.repeat(1, 3, 1)
    compute_normals(vertices[0], faces[0])

    vert_norms = torch.zeros_like(vertices).scatter_add(
        1, flat_face_idxs.unsqueeze(-1).long().repeat(1, 1, 3), flat_face_norms
    )
    vert_norms = torch_f.normalize(vert_norms, 2, -1)
    return face_norms, vert_norms
