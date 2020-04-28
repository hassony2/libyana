import numpy as np


def get_axis(axes, row_idx: int, col_idx: int, row_nb: int = 1, col_nb: int = 1):
    """
    Get matplotlib axis from array of axis returned by matplotlib
    subplots(row_nb, col_nb) given index of column and row,
    and total number of columns and rows

    Args:
        row_idx: idx or row to retriev
        col_idx: idx of column to retrieve
        row_nb: total number of rows
        col_nb: total number of cols
    """
    if col_nb == 1 and row_nb == 1:
        ax = axes
    elif col_nb == 1:
        ax = axes[row_idx]
    elif row_nb == 1:
        ax = axes[col_idx]
    else:
        ax = axes[row_idx, col_idx]
    return ax


def fig2np(fig):
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (h, w, 4)

    # canvas.tostring_argb give pixmap in ARGB mode.
    # Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf


def color_borders(
    ax, color=(1, 0, 0), locations=("top", "bottom", "right", "left"), hide_ticks=True
):
    for loc in locations:
        ax.spines[loc].set_color(color)
    if hide_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
