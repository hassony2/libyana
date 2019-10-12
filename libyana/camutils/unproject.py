import numpy as np


def recover_center3d(points3d, points2d, cam_intr):
    point_nb = points3d.shape[0]
    biases = []
    weights = []
    for point_idx in range(point_nb):
        weights.append(
            [cam_intr[0, 0], 0, cam_intr[0, 2] - points2d[point_idx][0]]
        )
        weights.append(
            [0, cam_intr[1, 1], cam_intr[1, 2] - points2d[point_idx][1]]
        )
        biases.append(
            -points3d[point_idx][0] * cam_intr[0, 0]
            - points3d[point_idx][2]
            * (cam_intr[0, 2] - points2d[point_idx][0])
        )
        biases.append(
            -points3d[point_idx][1] * cam_intr[1, 1]
            - points3d[point_idx][2]
            * (cam_intr[1, 2] - points2d[point_idx][1])
        )
    weights = np.stack(weights)
    biases = -np.array(biases)
    recovered_center, _, _, _ = np.linalg.lstsq(weights, -biases)
    return recovered_center
