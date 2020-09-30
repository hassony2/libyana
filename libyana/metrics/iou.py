def batch_mask_iou(ref, pred, eps=0.000001):
    ref = ref.float()
    pred = pred.float()
    if ref.max() > 1 or ref.min() < 0:
        raise ValueError(
            "Ref mask should have values in [0, 1], "
            f"not [{ref.min(), ref.max()}]"
        )
    if pred.max() > 1 or pred.min() < 0:
        raise ValueError(
            "Ref mask should have values in [0, 1], "
            f"not [{pred.min(), pred.max()}]"
        )

    inter = ref * pred
    union = (ref + pred).clamp(0, 1)
    ious = inter.sum(1).sum(1).float() / (union.sum(1).sum(1).float() + eps)
    return ious


def get_area(bbox):
    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    return area


def get_iou(bbox1, bbox2):
    """
    Args:
        bbox1/2 : bounding boxes in format x_min,y_min,x_max,y_max
    """

    bbox1 = [float(x) for x in bbox1]
    bbox2 = [float(x) for x in bbox2]

    overlap = [
        max(bbox1[0], bbox2[0]),
        max(bbox1[1], bbox2[1]),
        min(bbox1[2], bbox2[2]),
        min(bbox1[3], bbox2[3]),
    ]

    # check if there is overlap
    if overlap[2] - overlap[0] <= 0 or overlap[3] - overlap[1] <= 0:
        iou_area = 0
    else:
        area1 = get_area(bbox1)
        area2 = get_area(bbox2)
        intersection_area = get_area(overlap)
        union_area = area1 + area2 - intersection_area
        iou_area = intersection_area / union_area
    return iou_area
