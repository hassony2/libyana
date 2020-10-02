from matplotlib import pyplot as plt


def visualize_bbox(
    ax, box, color="r", label=None, alpha=1, linewidth=4, label_color=None
):
    """
    Args:
        box: [x_min, y_min, x_max, y_max]
    """
    coords = (box[0], box[1]), box[2] - box[0] + 1, box[3] - box[1] + 1
    ax.add_patch(
        plt.Rectangle(
            *coords,
            fill=False,
            edgecolor="g",
            linewidth=linewidth,
            color=color,
            alpha=alpha,
        )
    )
    if label_color is None:
        label_color = color
    ax.annotate(
        f"{label}",
        (box[0], box[1] + 4),
        color=label_color,
        weight="bold",
        alpha=alpha,
    )


def visualize_bboxes(
    ax, bboxes, color="r", labels=None, alpha=1, linewidth=4, label_color=None
):
    for box_idx, box in enumerate(bboxes):
        if labels is None:
            label = None
        else:
            label = labels[box_idx]
        if isinstance(color, str):
            col = color
        else:
            col = color[box_idx]
        visualize_bbox(
            ax,
            box,
            label=label,
            color=col,
            alpha=alpha,
            linewidth=linewidth,
            label_color=label_color,
        )
