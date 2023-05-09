import cv2
import numpy as np
import torch

class KuwaharaBlur:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "blur_radius": ("INT", {
                    "default": 3,
                    "min": 0,
                    "max": 31,
                    "step": 1
                }),
                "method": (["mean", "gaussian"],),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_kuwahara_filter"

    CATEGORY = "postprocessing/Filters"

    def apply_kuwahara_filter(self, image: np.ndarray, blur_radius: int, method: str):
        if blur_radius == 0:
            return (image,)

        out = torch.zeros_like(image)
        batch_size, height, width, channels = image.shape

        for b in range(batch_size):
            image = image[b].cpu().numpy() * 255.0
            image = image.astype(np.uint8)

            out[b] = torch.from_numpy(kuwahara(image, method=method, radius=blur_radius)) / 255.0

        return (out,)

def kuwahara(orig_img, method="mean", radius=3, sigma=None):
    if method == "gaussian" and sigma is None:
        sigma = -1

    image = orig_img.astype(np.float32, copy=False)
    avgs = np.empty((4, *image.shape), dtype=image.dtype)
    stddevs = np.empty((4, *image.shape[:2]), dtype=image.dtype)
    image_2d = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY).astype(image.dtype, copy=False)
    avgs_2d = np.empty((4, *image.shape[:2]), dtype=image.dtype)

    squared_img = image_2d ** 2

    if method == "mean":
        kxy = np.ones(radius + 1, dtype=image.dtype) / (radius + 1)
    elif method == "gaussian":
        kxy = cv2.getGaussianKernel(2 * radius + 1, sigma, ktype=cv2.CV_32F)
        kxy /= kxy[radius:].sum()
        klr = np.array([kxy[:radius+1], kxy[radius:]])
        kindexes = [[1, 1], [1, 0], [0, 1], [0, 0]]

    shift = [(0, 0), (0, radius), (radius, 0), (radius, radius)]

    for k in range(4):
        if method == "mean":
            kx, ky = kxy, kxy
        else:
            kx, ky = klr[kindexes[k]]
        cv2.sepFilter2D(image, -1, kx, ky, avgs[k], shift[k])
        cv2.sepFilter2D(image_2d, -1, kx, ky, avgs_2d[k], shift[k])
        cv2.sepFilter2D(squared_img, -1, kx, ky, stddevs[k], shift[k])
        stddevs[k] = stddevs[k] - avgs_2d[k] ** 2

    indices = np.argmin(stddevs, axis=0)
    filtered = np.take_along_axis(avgs, indices[None,...,None], 0).reshape(image.shape)

    return filtered.astype(orig_img.dtype)

NODE_CLASS_MAPPINGS = {
    "KuwaharaBlur": KuwaharaBlur
}
