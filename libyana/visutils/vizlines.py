from itertools import cycle
from libyana.conversions import npt


cycle_colors = ["c", "m", "y", "k", "r", "g", "b"]


def add_lines(ax, lines, over_lines=None, labels=None, overlay_alpha=0.6):
    colors = iter(cycle(cycle_colors))
    for line_idx, line in enumerate(lines):
        if labels is None:
            label = f"{line_idx}"
        else:
            label = labels[line_idx]
        color = next(colors)
        ax.plot(npt.numpify(line), label=label, c=color)
        if over_lines is not None:
            over_line = over_lines[line_idx]
            ax.plot(
                npt.numpify(over_line),
                "-",
                label=label,
                c=color,
                alpha=overlay_alpha,
            )
