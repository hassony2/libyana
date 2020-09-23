def visualize_corners_2d(
    ax, corners, joint_idxs=False, links=None, alpha=1, linewidth=2
):
    visualize_joints_2d(
        ax,
        corners,
        alpha=alpha,
        joint_idxs=joint_idxs,
        linewidth=linewidth,
        links=[
            [0, 1, 3, 2],
            [4, 5, 7, 6],
            [1, 5],
            [3, 7],
            [4, 0],
            [0, 2, 6, 4],
        ],
    )


def visualize_joints_2d(
    ax,
    joints,
    joint_idxs=True,
    links=None,
    alpha=1,
    scatter=True,
    linewidth=2,
    color=None,
    joint_labels=None,
    axis_equal=True,
):
    if links is None:
        links = [
            (0, 1, 2, 3, 4),
            (0, 5, 6, 7, 8),
            (0, 9, 10, 11, 12),
            (0, 13, 14, 15, 16),
            (0, 17, 18, 19, 20),
        ]
    # Scatter hand joints on image
    x = joints[:, 0]
    y = joints[:, 1]
    if scatter:
        ax.scatter(x, y, 1, "r")

    # Add idx labels to joints
    for row_idx, row in enumerate(joints):
        if joint_idxs:
            if joint_labels is None:
                joint_label = str(row_idx)
            else:
                joint_label = str(joint_labels[row_idx])
            ax.annotate(joint_label, (row[0], row[1]))
    _draw2djoints(
        ax, joints, links, alpha=alpha, linewidth=linewidth, color=color
    )
    if axis_equal:
        ax.axis("equal")


def _draw2djoints(ax, annots, links, alpha=1, linewidth=1, color=None):
    colors = ["r", "m", "b", "c", "g", "y", "b"]

    for finger_idx, finger_links in enumerate(links):
        for idx in range(len(finger_links) - 1):
            if color is not None:
                link_color = color[finger_idx]
            else:
                link_color = colors[finger_idx]
            _draw2dseg(
                ax,
                annots,
                finger_links[idx],
                finger_links[idx + 1],
                c=link_color,
                alpha=alpha,
                linewidth=linewidth,
            )


def _draw2dseg(ax, annot, idx1, idx2, c="r", alpha=1, linewidth=1):
    ax.plot(
        [annot[idx1, 0], annot[idx2, 0]],
        [annot[idx1, 1], annot[idx2, 1]],
        c=c,
        alpha=alpha,
        linewidth=linewidth,
    )
