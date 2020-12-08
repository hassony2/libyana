import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm

from libyana.conversions import npt
from libyana.visutils import vizmp


def conv2img(tensor):
    if len(tensor.shape) == 3:
        if tensor.shape[0] in [3, 4]:
            tensor = tensor.permute(1, 2, 0)
    tensor = npt.numpify(tensor)
    return tensor


def imagify(tensor, normalize_colors=True):
    tensor = npt.numpify(tensor)
    # Scale to [0, 1]
    if normalize_colors:
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    # Put channels last
    if tensor.ndim == 3 and tensor.shape[0] <= 4:
        tensor = tensor.transpose(1, 2, 0)
    if tensor.ndim == 3 and tensor.shape[2] == 1:
        tensor = tensor[:, :, 0]
    elif tensor.ndim == 3 and tensor.shape[2] < 3:
        tensor = np.concatenate(
            [tensor, 0.5 * np.ones_like(tensor)[:, :, : 3 - tensor.shape[2]]],
            2,
        )
    if tensor.ndim == 3 and (tensor.shape[2] != 4):
        tensor = tensor[:, :, :3]
    return tensor


def save2img(tensor, path="tmp.png"):
    img = conv2img(tensor)
    plt.imshow(img)
    plt.savefig(path)


def add_pointsrow(
    axes,
    tensors,
    overlay_list=None,
    overlay_colors=["c", "k", "b"],
    row_idx=0,
    row_nb=1,
    point_s=1,
    point_c="r",
    axis_equal=True,
    show_axis=True,
    alpha=1,
    over_alpha=1,
):
    point_nb = len(tensors)
    points = [conv2img(tens) for tens in tensors]
    for point_idx, point in enumerate(points):
        ax = vizmp.get_axis(
            axes,
            row_idx=row_idx,
            col_idx=point_idx,
            row_nb=row_nb,
            col_nb=point_nb,
        )
        pts = npt.numpify(points[point_idx])
        if point_c == "rainbow":
            pt_nb = pts.shape[0]
            point_c = cm.rainbow(np.linspace(0, 1, pt_nb))
        ax.scatter(pts[:, 0], pts[:, 1], s=point_s, c=point_c, alpha=alpha)
        if overlay_list is not None:
            for overlay, over_color in zip(overlay_list, overlay_colors):
                over_pts = npt.numpify(overlay[point_idx])
                ax.scatter(
                    over_pts[:, 0],
                    over_pts[:, 1],
                    s=point_s,
                    c=over_color,
                    alpha=over_alpha,
                )
        if axis_equal:
            ax.axis("equal")
        if not show_axis:
            ax.axis("off")


def viz_pointsrow(
    tensors,
    path="tmp.png",
    points=None,
    fig_height=3,
    show_axis=True,
    point_s=1,
    point_c="r",
    overlay_list=None,
    overlay_colors=("c", "k", "b"),
    title=None,
):
    img_nb = len(tensors)
    fig, axes = plt.subplots(
        1, img_nb, figsize=(int(img_nb * fig_height), fig_height)
    )
    add_pointsrow(
        axes,
        tensors,
        row_idx=0,
        row_nb=1,
        point_s=point_s,
        point_c=point_c,
        show_axis=show_axis,
        overlay_list=overlay_list,
        overlay_colors=overlay_colors,
    )
    if title is not None:
        fig.suptitle(title)
    fig.savefig(path)
    plt.show()


def add_imgrow(
    axes,
    tensors,
    row_idx=0,
    row_nb=1,
    overlays=None,
    points=None,
    over_alpha=0.5,
    show_axis=False,
    point_s=1,
    point_c="r",
    overval=1,
):
    img_nb = len(tensors)
    imgs = [conv2img(tens) for tens in tensors]
    for img_idx, img in enumerate(imgs):
        ax = vizmp.get_axis(
            axes,
            row_idx=row_idx,
            col_idx=img_idx,
            row_nb=row_nb,
            col_nb=img_nb,
        )
        if points is not None:
            pts = npt.numpify(points[img_idx])
            if pts is not None:
                ax.scatter(pts[:, 0], pts[:, 1], s=point_s, c=point_c)
        if overlays is not None:
            overlay_img = conv2img(overlays[img_idx])
            if overlay_img.ndim == 2:
                overlay_mask = (overlay_img != overval)[:, :]
            else:
                overlay_mask = (overlay_img.sum(2) != overval * 3)[
                    :, :, np.newaxis
                ]
            over_img = (
                overlay_mask * img + (1 - overlay_mask) * img * over_alpha
            )
            ax.imshow(over_img)
        else:
            ax.imshow(img)

        if not show_axis:
            ax.axis("off")


def viz_imgrow(
    tensors,
    path="tmp.png",
    points=None,
    fig_height=3,
    show_axis=False,
    point_s=1,
    point_c="r",
    overlays=None,
    overval=1,
    over_alpha=0.5,
    title=None,
    viz_nb=None,
):
    if viz_nb is not None:
        img_idxs = np.linspace(0, len(tensors) - 1, viz_nb).astype(np.int)
        tensors = [tensors[idx] for idx in img_idxs]
        if overlays is not None:
            overlays = [overlays[idx] for idx in img_idxs]
    img_nb = len(tensors)
    fig, axes = plt.subplots(
        1, img_nb, figsize=(int(img_nb * fig_height), fig_height)
    )
    add_imgrow(
        axes,
        tensors,
        row_idx=0,
        row_nb=1,
        overlays=overlays,
        points=points,
        over_alpha=over_alpha,
        overval=overval,
        point_s=point_s,
        point_c=point_c,
        show_axis=show_axis,
    )
    if title is not None:
        fig.suptitle(title)
    fig.savefig(path)
    plt.show()


def viz_imgrid(tensors_list, path="tmp.png", fig_height=3, show_axis=False):
    row_nb = len(tensors_list)
    img_nb = len(tensors_list[0])
    fig, axes = plt.subplots(
        row_nb,
        img_nb,
        figsize=(int(img_nb * fig_height), int(row_nb * fig_height)),
    )
    for row_idx, row_imgs in enumerate(tensors_list):
        imgs = [conv2img(tens) for tens in row_imgs]
        for img_idx, img in enumerate(imgs):
            ax = vizmp.get_axis(axes, row_idx, img_idx, row_nb, img_nb)
            ax.imshow(img)
            if show_axis:
                ax.axis("off")
    if path:
        fig.savefig(path)
    return fig
