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
