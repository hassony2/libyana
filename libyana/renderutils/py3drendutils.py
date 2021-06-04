import numpy as np
import torch
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    BlendParams,
    PerspectiveCameras,
    PointLights,
    DirectionalLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    HardPhongShader,
    HardFlatShader,
    SoftSilhouetteShader,
)
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.renderer.mesh import textures
from libyana.renderutils.blending import soft_feature_blending


def get_colors(verts, color=(0.53, 0.53, 0.8)):
    colors = (torch.from_numpy(np.array(color)).view(1, 1, 3).to(
        verts.device).float().repeat(verts.shape[0], verts.shape[1], 1))
    return colors


def batch_render(
    verts,
    faces,
    faces_per_pixel=10,
    K=None,
    rot=None,
    trans=None,
    colors=None,
    color=(0.53, 0.53, 0.8),  # light_purple
    ambient_col=0.5,
    specular_col=0.2,
    diffuse_col=0.3,
    face_colors=None,
    # color = (0.74117647, 0.85882353, 0.65098039),  # light_blue
    image_sizes=None,
    out_res=512,
    bin_size=0,
    shading="soft",
    mode="rgb",
    blend_gamma=1e-4,
    blend_sigma=1e-4,
    min_depth=None,
):
    device = torch.device("cuda:0")
    K = K.to(device)
    width, height = image_sizes[0]
    out_size = int(max(image_sizes[0]))
    raster_settings = RasterizationSettings(
        image_size=out_size,
        blur_radius=0.0,
        faces_per_pixel=faces_per_pixel,
        bin_size=bin_size,
    )

    fx = K[:, 0, 0]
    fy = K[:, 1, 1]
    focals = torch.stack([fx, fy], 1)
    px = K[:, 0, 2]
    py = K[:, 1, 2]
    principal_point = torch.stack([width - px, height - py], 1)
    if rot is None:
        rot = torch.eye(3).unsqueeze(0).to(device)
    if trans is None:
        trans = torch.zeros(3).unsqueeze(0).to(device)
    cameras = PerspectiveCameras(
        device=device,
        focal_length=focals,
        principal_point=principal_point,
        image_size=[(out_size, out_size) for _ in range(len(verts))],
        R=rot,
        T=trans,
    )
    if mode == "rgb":
        
        lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
        lights = DirectionalLights(
            device=device,
            direction=((0.6, -0.6, -0.6), ),
            ambient_color=((ambient_col, ambient_col, ambient_col), ),
            diffuse_color=((diffuse_col, diffuse_col, diffuse_col), ),
            specular_color=((specular_col, specular_col, specular_col), ),
        )
        if shading == "soft":
            blend_params = BlendParams(sigma=blend_sigma, gamma=blend_gamma)
            shader = SoftPhongShader(device=device, cameras=cameras, lights=lights, blend_params=blend_params)
        elif shading == "hard":
            shader = HardPhongShader(device=device, cameras=cameras, lights=lights)
        elif shading == "flat":
            shader = HardFlatShader(device=device, cameras=cameras, lights=lights)
        else:
            raise ValueError(f"Shading {shading} for mode rgb not in [sort|hard]")
    elif mode == "silh":
        blend_params = BlendParams(sigma=blend_sigma, gamma=blend_gamma)
        shader = SoftSilhouetteShader(blend_params=blend_params)
    elif shading == "faceidx":
        shader = FaceIdxShader()
    elif (mode == "facecolor") and (shading == "hard"):
        shader = FaceColorShader(face_colors=face_colors)
    elif (mode == "facecolor") and (shading == "soft"):
        shader = SoftFaceColorShader(face_colors=face_colors,
                                     blend_gamma=blend_gamma,
                                     blend_sigma=blend_sigma)
    else:
        raise ValueError(
            f"Unhandled mode {mode} and shading {shading} combination")

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras,
                                  raster_settings=raster_settings),
        shader=shader,
    )
    if min_depth is not None:
        verts = torch.cat([verts[:, :, :2], verts[:, :, 2:].clamp(min_depth)],
                          2)
    if mode == "rgb":
        if colors is None:
            colors = get_colors(verts, color)
        tex = textures.TexturesVertex(verts_features=colors)

        meshes = Meshes(verts=verts, faces=faces, textures=tex)
    elif mode in ["silh", "facecolor"]:
        meshes = Meshes(verts=verts, faces=faces)
    else:
        raise ValueError(f"Render mode {mode} not in [rgb|silh]")

    square_images = renderer(meshes, cameras=cameras)
    height_off = int(width - height)
    # from matplotlib import pyplot as plt
    # plt.imshow(square_images.cpu()[0, :, :, 0])
    # plt.savefig("tmp.png")
    images = torch.flip(square_images, (1, 2))[:, height_off:]
    return images


class SoftFaceColorShader(torch.nn.Module):
    def __init__(self, face_colors=None, device="cpu", blend_gamma=1e-2, blend_sigma=1e-4):
        super().__init__()
        batch_s, face_nb, color_nb = face_colors.shape
        vert_face_colors = face_colors.unsqueeze(2).repeat(1, 1, 3, 1)
        self.face_colors = vert_face_colors.view(batch_s * face_nb, 3,
                                                 color_nb).float()
        self.blend_gamma = blend_gamma
        self.blend_sigma = blend_sigma

    def forward(self, fragments, meshes, **kwargs):
        colors = interpolate_face_attributes(fragments.pix_to_face,
                                             fragments.bary_coords,
                                             self.face_colors)
        blend_params = BlendParams(sigma=self.blend_sigma, gamma=self.blend_gamma)
        imgs = soft_feature_blending(colors,
                                     fragments,
                                     blend_params=blend_params)
        return imgs


class FaceColorShader(torch.nn.Module):
    def __init__(self, face_colors=None, device="cpu"):
        super().__init__()
        batch_s, face_nb, color_nb = face_colors.shape
        vert_face_colors = face_colors.unsqueeze(2).repeat(1, 1, 3, 1)
        self.face_colors = vert_face_colors.view(batch_s * face_nb, 3,
                                                 color_nb).float()

    def forward(self, fragments, meshes, **kwargs):
        colors = interpolate_face_attributes(fragments.pix_to_face,
                                             fragments.bary_coords,
                                             self.face_colors)
        return colors[:, :, :, 0]


class FaceIdxShader(torch.nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()

    def forward(self, fragments, meshes, **kwargs):
        return fragments.pix_to_face
