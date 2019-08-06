import random

from libyana.transformutils import torchfn


def get_color_params(brightness=0, contrast=0, saturation=0, hue=0):
    if brightness > 0:
        bright_factor = random.uniform(max(0, 1 - brightness), 1 + brightness)
    else:
        bright_factor = None

    if contrast > 0:
        contrast_factor = random.uniform(max(0, 1 - contrast), 1 + contrast)
    else:
        contrast_factor = None

    if saturation > 0:
        sat_factor = random.uniform(max(0, 1 - saturation), 1 + saturation)
    else:
        sat_factor = None

    if hue > 0:
        hue_factor = random.uniform(-hue, hue)
    else:
        hue_factor = None
    return bright_factor, contrast_factor, sat_factor, hue_factor


def apply_jitter(img, brightness=0, contrast=0, saturation=0, hue=0):
    # Create img transform function sequence
    img_transforms = []
    if brightness is not None:
        img_transforms.append(
            lambda img: torchfn.adjust_brightness(img, brightness)
        )
    if saturation is not None:
        img_transforms.append(
            lambda img: torchfn.adjust_saturation(img, saturation)
        )
    if hue is not None:
        img_transforms.append(lambda img: torchfn.adjust_hue(img, hue))
    if contrast is not None:
        img_transforms.append(
            lambda img: torchfn.adjust_contrast(img, contrast)
        )
    random.shuffle(img_transforms)

    jittered_img = img
    for func in img_transforms:
        jittered_img = func(jittered_img)
    return jittered_img


def color_jitter(img, brightness=0, contrast=0, saturation=0, hue=0):
    brightness, contrast, saturation, hue = get_color_params(
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue,
    )
    jittered_img = apply_jitter(
        img,
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue,
    )
    return jittered_img
