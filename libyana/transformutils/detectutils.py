import numpy as np


def make_square_bbox(box, img_size=None):
    box_scale = 1.5 * max(box[2] - box[0], box[3] - box[1])
    center_x, center_y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    if img_size is not None:
        box_scale = min(box_scale, min(img_size))
        if center_x < box_scale / 2:
            center_x = box_scale / 2
        if center_y < box_scale / 2:
            center_y = box_scale / 2
        if center_x + box_scale / 2 > img_size[0]:
            center_x = img_size[0] - box_scale / 2
        if center_y + box_scale / 2 > img_size[1]:
            center_y = img_size[1] - box_scale / 2
    box_center = np.array([center_x, center_y])
    square_box = (
        box_center[0] - box_scale / 2,
        box_center[1] - box_scale / 2,
        box_center[0] + box_scale / 2,
        box_center[1] + box_scale / 2,
    )
    return square_box
