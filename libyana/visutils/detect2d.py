from matplotlib import pyplot as plt


def visualize_bbox(ax, box, color="r", label=None, alpha=1, linewidth=4):
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
    ax.annotate(
        f"{label}", (box[0], box[1] + 4), color=color, weight="bold", alpha=alpha
    )


def visualize_bboxes(ax, bboxes, color="r", labels=None, alpha=1, linewidth=4):
    for box_idx, box in enumerate(bboxes):
        if labels is None:
            label = None
        else:
            label = labels[box_idx]
        visualize_bbox(
            ax, box, label=label, color="r", alpha=alpha, linewidth=linewidth
        )
