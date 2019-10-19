import torch


def vertex_textures(
    faces, vertex_colors, texture_size=2, with_artefacts=False
):
    """
    Create textures for neural_renderer from vertex colors

    Args:
        faces (nb_faces, 3): contains the indexes of vertex for each face
        vertex_colors (vertex_nb, 3)
        with_artefacts (bool): Create artefacts that highlight vertices
    """
    assert faces.dim() == 2, f"expected shape (face_nb, 3), got {faces.shape}"
    assert (
        vertex_colors.dim() == 2
    ), f"expected shape (face_nb, 3), got {faces.shape}"
    assert (
        faces.shape[1] == 3
    ), f"expected shape (face_nb, 3), got {faces.shape}"
    assert (
        vertex_colors.shape[1] == 3
    ), f"expect shape (face_nb, 3), got {faces.shape}"
    textures = torch.zeros(
        faces.shape[0], texture_size, texture_size, texture_size, 3
    )
    f1_colors = vertex_colors[faces[:, 0]]
    f2_colors = vertex_colors[faces[:, 1]]
    f3_colors = vertex_colors[faces[:, 2]]
    if texture_size > 1:
        for tex_idx1 in range(texture_size):
            for tex_idx2 in range(texture_size):
                for tex_idx3 in range(texture_size):
                    sum_idxs = tex_idx1 + tex_idx2 + tex_idx3
                    if sum_idxs:
                        if with_artefacts:
                            textures[:, tex_idx1, tex_idx2, tex_idx3] = (
                                tex_idx1 * f1_colors
                                + tex_idx2 * f2_colors
                                + tex_idx3 * f3_colors
                            ) / sum_idxs
                        else:
                            textures[:, tex_idx1, tex_idx2, tex_idx3] = (
                                tex_idx1 * f1_colors
                                + tex_idx2 * f2_colors
                                + tex_idx3 * f3_colors
                            )
    else:
        textures[:, 0, 0, 0] = f1_colors
    return textures


def batch_vertex_textures(
    faces, vertex_colors, texture_size=2, with_artefacts=False
):
    """
    Create textures for neural_renderer from vertex colors
    Args:
        faces (batch_size, nb_faces, 3): contains the indexes of vertex for each face
        vertex_colors (batch_size, vertex_nb, 3)
    """
    assert faces.dim() == 3, f"expected shape (face_nb, 3), got {faces.shape}"
    assert (
        vertex_colors.dim() == 3
    ), f"expected shape (batch_size, face_nb, 3), got {faces.shape}"
    assert (
        faces.shape[2] == 3
    ), f"expected shape (batch_size, face_nb, 3), got {faces.shape}"
    assert (
        vertex_colors.shape[2] == 3
    ), f"expect shape (face_nb, 3), got {faces.shape}"
    textures = vertex_colors.new_zeros(
        faces.shape[0],
        faces.shape[1],
        texture_size,
        texture_size,
        texture_size,
        3,
    )
    f1_colors = torch.gather(
        vertex_colors, 1, faces[:, :, 0:1].repeat(1, 1, 3).long()
    )
    f2_colors = torch.gather(
        vertex_colors, 1, faces[:, :, 1:2].repeat(1, 1, 3).long()
    )
    f3_colors = torch.gather(
        vertex_colors, 1, faces[:, :, 2:].repeat(1, 1, 3).long()
    )
    if texture_size > 1:
        for tex_idx1 in range(texture_size):
            for tex_idx2 in range(texture_size):
                for tex_idx3 in range(texture_size):
                    sum_idxs = tex_idx1 + tex_idx2 + tex_idx3
                    if sum_idxs:
                        if with_artefacts:
                            textures[:, :, tex_idx1, tex_idx2, tex_idx3] = (
                                tex_idx1 * f1_colors
                                + tex_idx2 * f2_colors
                                + tex_idx3 * f3_colors
                            ) / sum_idxs
                        else:
                            textures[:, :, tex_idx1, tex_idx2, tex_idx3] = (
                                tex_idx1 * f1_colors
                                + tex_idx2 * f2_colors
                                + tex_idx3 * f3_colors
                            )
    else:
        textures[:, :, 0, 0, 0] = f1_colors
    return textures
