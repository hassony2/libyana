import torch
from matplotlib import pyplot as plt

plt.switch_backend("agg")


def tri_area(v):
    return 0.5 * torch.norm(
        torch.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), 2, 1
    )


def points_from_mesh(faces, vertices, vertex_nb=600, show_cloud=False):
    verts_faces = vertices[faces.long()]
    areas = tri_area(verts_faces)

    proba = areas / areas.sum()
    rand_idxs = torch.multinomial(proba, vertex_nb, True)

    # Randomly pick points on triangles
    u = torch.rand(vertex_nb, 1)
    v = torch.rand(vertex_nb, 1)

    # Force bernouilli couple to be picked on a half square
    out = u + v > 1
    u[out] = 1 - u[out]
    v[out] = 1 - v[out]

    rand_tris = vertices[faces[rand_idxs]]
    points = (
        rand_tris[:, 0]
        + u * (rand_tris[:, 1] - rand_tris[:, 0])
        + v * (rand_tris[:, 2] - rand_tris[:, 0])
    )

    if show_cloud:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(points[:, 0], points[:, 1], s=2, c="b")
        ax.scatter(vertices[:, 0], vertices[:, 1], s=2, c="r")
        ax._axis3don = False
        plt.savefig("tmp.png")
    return points
