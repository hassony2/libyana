import numpy as np

from libyana.conversions import npt
from libyana.verify import checkshape


def batch_proj2d(verts, camintr, camextr=None):
    # Project 3d vertices on image plane
    if camextr is not None:
        verts = camextr.bmm(verts.transpose(1, 2)).transpose(1, 2)
    verts_hom2d = camintr.bmm(verts.transpose(1, 2)).transpose(1, 2)
    proj_verts2d = verts_hom2d[:, :, :2] / verts_hom2d[:, :, 2:]
    return proj_verts2d


def homogenify_np(pts):
    pts = npt.numpify(pts)
    if (pts.ndim != 2) | (pts.shape[1] not in [2, 3]):
        pass
        raise ValueError(
            f"{pts.shape} is not valid (point_nb, 2/3) points shape"
        )
    hom3d = np.concatenate([pts, np.ones([np.array(pts).shape[0], 1])], 1)
    return hom3d


def proj2d(verts, camintr, camextr=None, rot=None, trans=None):
    if camextr is not None:
        if rot is not None:
            raise ValueError(
                "camextr and rot should not be set simultaneously."
            )
        if trans is not None:
            raise ValueError(
                "camextr and trans should not be set simultaneously."
            )

    verts = npt.numpify(verts)
    camintr = npt.numpify(camintr)
    if camextr is not None:
        camextr = npt.numpify(camextr)
    if rot is not None:
        rot = npt.numpify(rot)
        checkshape.check_shape(rot, (3, 3), name="rot")
    if trans is not None:
        trans = npt.numpify(trans)
        if trans.ndim == 1:
            trans = trans[None, :]
        checkshape.check_shape(trans, (1, 3), name="rot")
    checkshape.check_shape(verts, (-1, 3), name="verts")
    if (camintr.ndim != 2) & (camintr.shape[1] == 3) & (camintr.shape[0] == 3):
        raise ValueError(f"{camintr.shape} is not valid (3, 3) camintr shape")
    if rot is not None:
        verts = rot.dot(verts.transpose()).transpose()
    if trans is not None:
        verts = trans + verts
    if camextr is not None:
        checkshape.check_shape(camextr, (4, 4), name="camextr")
        hom3d = np.homogenify_np(verts)
        hom3d = camextr.dot(hom3d.transpose()).transpose()
        verts = hom3d[:, :3] / hom3d[:, 3:]

    hom2d = camintr.dot(verts.transpose()).transpose()
    pts2d = hom2d[:, :2] / hom2d[:, 2:]
    return pts2d
