import numpy as np
import cv2
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
        tensor_image = image.numpy()[0]
        gray_image = (cv2.cvtColor(tensor_image, cv2.COLOR_BGR2GRAY) * 255).astype(np.uint8)
        canny = cv2.Canny(gray_image, lower_threshold, upper_threshold)
        tensor = torch.from_numpy(canny).unsqueeze(0)
        return (tensor,)

NODE_CLASS_MAPPINGS = {
    "CannyEdgeDetection": CannyEdgeDetection
}