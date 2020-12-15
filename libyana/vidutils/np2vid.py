import cv2
from moviepy import editor


def make_video(imgs, video_paths, fps=2, resize_factor=1, verbose=True):
    if resize_factor != 1:
        resize_imgs = []
        for img in imgs:
            img = cv2.resize(
                img,
                (
                    int(img.shape[1] * resize_factor),
                    int(img.shape[0] * resize_factor),
                ),
            )
            resize_imgs.append(img)
        imgs = resize_imgs
    clip = editor.ImageSequenceClip(imgs, fps=fps)
    if isinstance(video_paths, str):
        video_paths = [video_paths]
    for video_path in video_paths:
        if video_path.endswith(".gif"):
            clip.write_gif(str(video_path))
        else:
            clip.write_videofile(str(video_path))
        if verbose:
            print(f"Saved video to {video_path}")
