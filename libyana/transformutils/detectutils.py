import numpy as np


def out_of_bound_bboxes(boxes, img_size, margin=0):
    out_of_bounds = []
    for box in boxes:
        center_x, center_y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        box_scale = max(box[2] - box[0], box[3] - box[1])
        if (
            (center_x < margin * box_scale)
            or (center_x > img_size[0] + margin * box_scale)
            or (center_y < margin * box_scale)
            or (center_y > img_size[1] + margin * box_scale)
        ):
            out_of_bounds.append(True)
        else:
            out_of_bounds.append(False)
    return out_of_bounds


def make_square_bbox(box, img_size=None):
    box_scale = 1.5 * max(box[2] - box[0], box[3] - box[1])
    center_x, center_y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    if img_size is not None:
        box_scale = min(box_scale, min(img_size))
        # Snap to boundaries while keeping square
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
