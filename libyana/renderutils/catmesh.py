import torch


def batch_cat_meshes(verts, faces, textures=None):
    assert len(verts) == len(
        faces
    ), f"Different number of vertices{len(verts)} and faces {len(faces)}"
    cur_verts = verts[0]
    cur_faces = faces[0]
    if textures is None:
        cur_textures = None
    else:
        cur_textures = textures[0]
    for mesh_idx in range(1, len(verts)):
        if textures is None:
            cat_textures = None
        else:
            cat_textures = [cur_textures, textures[mesh_idx]]

        cur_verts, cur_faces, cur_textures = cat_2meshes(
            [cur_verts, verts[mesh_idx]],
            [cur_faces, faces[mesh_idx]],
            cat_textures,
        )
    return cur_verts, cur_faces, cur_textures


def cat_2meshes(verts, faces, textures=None):
    """
    Concatenates lists of two batched meshes in neural_renderer format

    Args:
        verts list[batch_size, vertex_nb, 3]: list of vertices of objects
        faces list[batch_size, face_nb, 3]: list of faces of objects
        textures list[batch_size, face_nb, tex_size, tex_size, tex_size, 3]:
            list of textures of objects
    """
    assert (
        len(verts) == 2
    ), f"cat_2meshes got list of {len(verts)} verts, expected 2"
    assert (
        len(faces) == 2
    ), f"cat_2meshes got list of {len(faces)} faces, expected 2"
    assert (
        faces[0].dim() == 3
    ), f"expected shape (face_nb, 3), got {faces.shape}"
    assert (
        faces[1].dim() == 3
    ), f"expected shape (face_nb, 3), got {faces.shape}"
    assert faces[0].shape[2] == 3, (
        f"expected shape (batch_size, face_nb, 3)," f"got {faces.shape}"
    )
    assert faces[1].shape[2] == 3, (
        f"expected shape (batch_size, face_nb, 3)," f"got {faces.shape}"
    )
    if textures is not None:
        assert len(textures) == 2
    all_verts = torch.cat([verts[0], verts[1]], 1)
    all_faces = torch.cat([faces[0], faces[1] + verts[0].shape[1]], 1)
    if textures is not None:
        all_textures = torch.cat([textures[0], textures[1]], 1)
    else:
        all_textures = None
    return all_verts, all_faces, all_textures
