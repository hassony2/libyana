import torch


def vertex_textures(faces, vertex_colors, texture_size=2):
    """
    Create textures for neural_renderer from vertex colors

    Args:
        faces (nb_faces, 3): contains the indexes of vertex for each face
        vertex_colors (vertex_nb, 3)
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
                        textures[:, tex_idx1, tex_idx2, tex_idx3] = (
                            tex_idx1 / sum_idxs * f1_colors
                            + tex_idx2 / sum_idxs * f2_colors
                            + tex_idx3 / sum_idxs * f3_colors
                        )
    else:
        textures[:, 0, 0, 0] = f1_colors
    return textures
