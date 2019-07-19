import numpy as np


def make_square_bbox(box):
    box_scale = 1.5 * max(box[2] - box[0], box[3] - box[1])
    box_center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
    square_box = (
        box_center[0] - box_scale / 2,
        box_center[1] - box_scale / 2,
        box_center[0] + box_scale / 2,
        box_center[1] + box_scale / 2,
    )
    return square_box
