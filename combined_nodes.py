import cv2
import numpy as np
import torch


class CannyEdgeDetection:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "lower_threshold": ("INT", {
                    "default": 100,
                    "min": 0,
                    "max": 500,
                    "step": 10
                }),
                "upper_threshold": ("INT", {
                    "default": 200,
                    "min": 0,
                    "max": 500,
                    "step": 10
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "canny"

    CATEGORY = "postprocessing"

    def canny(self, image: torch.Tensor, lower_threshold: int, upper_threshold: int):
        batch_size, height, width, _ = image.shape
        result = torch.zeros(batch_size, height, width)

        for b in range(batch_size):
            tensor_image = image[b].numpy().copy()
            gray_image = (cv2.cvtColor(tensor_image, cv2.COLOR_RGB2GRAY) * 255).astype(np.uint8)
            canny = cv2.Canny(gray_image, lower_threshold, upper_threshold)
            tensor = torch.from_numpy(canny)
            result[b] = tensor

        return (result,)

class GaussianBlur:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "kernel_size": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 31,
                    "step": 1
                }),
                "sigma": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "blur"

    CATEGORY = "postprocessing"

    def blur(self, image: torch.Tensor, kernel_size: int, sigma: float):
        batch_size, height, width, _ = image.shape
        result = torch.zeros_like(image)

        for b in range(batch_size):
            tensor_image = image[b].numpy()
            blurred = cv2.GaussianBlur(tensor_image, (kernel_size, kernel_size), sigma)
            tensor = torch.from_numpy(blurred).unsqueeze(0)
            result[b] = tensor

        return (result,)

NODE_CLASS_MAPPINGS = {
    "CannyEdgeDetection": CannyEdgeDetection,
    "GaussianBlur": GaussianBlur,
}
