import imageio
import torch
import torchvision
from skimage import io
import matplotlib
from matplotlib import pyplot as plt
from torchvision.transforms import Resize, Normalize
import cv2
import numpy as np
import datetime

def get_datetime_str(style='dt'):
    cur_time = datetime.datetime.now()

    date_str = cur_time.strftime('%y_%m_%d_')
    time_str = cur_time.strftime('%H_%M_%S')

    if style == 'data':
        return date_str
    elif style == 'time':
        return time_str
    else:
        return date_str + time_str

def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """
    attention!  this function is not implemented by me!
    from https://github.com/jacobgil/pytorch-grad-cam

    This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.
    :param img: The base image in RGB or BGR format.  H, D, C and all value in [0, 1]
    :param mask: The cam mask.    H, D and all value in [0, 1]
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def show_cam(image: torch.tensor or np.ndarray or None,
             grad: torch.tensor or np.ndarray,
             if_save_image=True):
    if isinstance(image, torch.tensor):
        image = image.squeeze().numpy()
    if isinstance(grad, torch.tensor):
        grad = grad.squeeze().numpy()
    if image.shape[0] == 3:
        image = image.transpose(1, 2, 0)
    assert len(grad.shape) < 3, 'grad should be H,W, if not, please use mean()'

    if image is None:
        temp = grad / np.max(grad)
        temp = temp.permute(1, 2, 0).numpy()
        plt.imshow(temp)
    else:
        im = show_cam_on_image(image, grad)
        plt.imshow(im)
    plt.show()
    if if_save_image:
        plt.savefig(get_datetime_str() + ".png")
        
