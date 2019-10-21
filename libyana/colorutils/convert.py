import torch


def rgb2hsv(tens):
    """
    Expects inputs of shape [batch_size, H, W, 3] in range [0, 1]
    """
    orig_shape = tens.shape
    tens = tens.view(-1, 3)
    mx = tens.max(-1)[0]
    mn = tens.min(-1)[0]
    df = mx - mn
    h_c = torch.zeros_like(df)
    h_c[mx == tens[:, 0]] = (60 * (tens[:, 1] - tens[:, 2]) / df + 360 % 360)[
        mx == tens[:, 0]
    ]
    h_c[mx == tens[:, 1]] = (60 * (tens[:, 2] - tens[:, 0]) / df + 120 % 360)[
        mx == tens[:, 1]
    ]
    h_c[mx == tens[:, 2]] = (60 * (tens[:, 0] - tens[:, 1]) / df + 240 % 360)[
        mx == tens[:, 2]
    ]
    s_c = torch.zeros_like(df)
    s_c[mx != 0] = (df / mx)[mx != 0]
    return torch.stack(
        [
            h_c.view(orig_shape[:-1]) / 360,
            s_c.view(orig_shape[:-1]),
            mx.view(orig_shape[:-1]),
        ],
        -1,
    )
