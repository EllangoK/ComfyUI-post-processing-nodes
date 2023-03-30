import numpy as np
import cv2
import torch

class Sharpen:
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
                "alpha": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "sharpen"

    CATEGORY = "postprocessing"

    def sharpen(self, image, alpha, kernel_size):
        tensor_img = image.numpy()[0]

        kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) * -1
        center = kernel_size // 2
        kernel[center, center] = kernel_size**2
        kernel *= alpha

        sharpened = cv2.filter2D(tensor_img, -1, kernel)

        tensor = torch.from_numpy(sharpened).unsqueeze(0)
        return (tensor,)

NODE_CLASS_MAPPINGS = {
    "Sharpen": Sharpen
}