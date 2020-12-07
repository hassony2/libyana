import torch
from pytorch3d.renderer.blending import BlendParams


def soft_feature_blending(colors,
                          fragments,
                          blend_params=None,
                          znear: float = 0.01,
                          zfar: float = 10) -> torch.Tensor:
    """
    Returns:
        Rendered features: (N, H, W, F)
    """
    blend_params = BlendParams() if blend_params is None else blend_params
    eps = 1e-10  # Weight for background
    N, H, W, K = fragments.pix_to_face.shape
    # number of feature channels
    # C = colors.shape[-1]

    mask = fragments.pix_to_face >= 0
    prob_map = torch.sigmoid(-fragments.dists / blend_params.sigma) * mask
    z_inv = (zfar - fragments.zbuf) / (zfar - znear) * mask
    # pyre-fixme[16]: `Tuple` has no attribute `values`.
    # pyre-fixme[6]: Expected `Tensor` for 1st param but got `float`.
    z_inv_max = torch.max(z_inv, dim=-1).values[..., None].clamp(min=eps)
    # pyre-fixme[6]: Expected `Tensor` for 1st param but got `float`.
    z_prob = torch.exp((z_inv - z_inv_max) / blend_params.gamma)
    z_prob = z_prob / (z_prob.sum(-1, keepdim=True).clamp(eps, 1))

    # For each face, compute the soft assignment score by taking into account xy dist
    # and relative z distance
    # (among different faces assigned to specific xy pixel location)
    colored = 1 - torch.prod(1 - (prob_map * z_prob).unsqueeze(-1) * colors,
                             dim=-2)
    alpha = 1 - torch.prod((1.0 - prob_map), dim=-1)
    return torch.cat([colored, alpha.unsqueeze(-1)], -1)
