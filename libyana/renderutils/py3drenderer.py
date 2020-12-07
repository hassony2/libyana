import torch
from libyana.renderutils import py3drendutils


class Renderer:
    def __init__(self, ):
        super().__init__()

    def render_batch(self, obj_infos, cam_infos):
        verts = obj_infos["verts"].float().cuda()
        faces = obj_infos["faces"].cuda()
        K = cam_infos["K"].float()
        TCO = cam_infos["TCO"].float().cuda()
        verts = torch.bmm(TCO[:, :3, :3], verts.permute(0, 2,
                                                        1)).permute(0, 2, 1)
        verts = verts + TCO[:, :3, 3].unsqueeze(1)
        image_sizes = cam_infos["resolution"]
        images = py3drendutils.batch_render(verts,
                                            faces,
                                            K=K,
                                            image_sizes=image_sizes)
        return {"rgb": images}
