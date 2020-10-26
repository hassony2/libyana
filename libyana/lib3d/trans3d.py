from pytorch3d import transforms as py3dt

from libyana.conversions import npt


def rot_points(points, centers=None, axisang=(0, 1, 1)):
    """
    Rotate points around centers

    Args:
        points (torch.Tensor): (batch_size, point_nb, 3) points to rotate
    """
    points = npt.tensorify(points)
    if points.dim() != 3 or (points.shape[2] != 3):
        raise ValueError(
            "Expected batch of vertices in format (batch_size, vert_nb, 3)"
            f" but got {points.shape}"
        )

    if centers is None:
        centers = points.mean(1)
    else:
        centers = npt.tensorify(centers, points.device)

    if centers.dim() == 2:
        centers = centers.unsqueeze(1)
    points_c = points - centers
    rot_mats = py3dt.so3_exponential_map(
        points.new(axisang).unsqueeze(0)
    ).view(1, 3, 3)
    points_cr = (
        rot_mats.repeat(points.shape[0], 1, 1)
        .bmm(points_c.transpose(1, 2))
        .transpose(1, 2)
    )
    points_final = points_cr + centers
    return points_final
