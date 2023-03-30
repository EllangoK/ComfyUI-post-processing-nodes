import numpy as np
import cv2
import torch

class KMeansQuantize:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "colors": ("INT", {
                    "default": 16,
                    "min": 1,
                    "max": 256,
                    "step": 1
                }),
                "precision": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 100,
                    "step": 1
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "kmeans_quantize"

    CATEGORY = "postprocessing"

    def kmeans_quantize(self, image: torch.Tensor, colors: int, precision: int):
        tensor_image = image.numpy()[0].astype(np.float32)
        img = tensor_image

        height, width, c = img.shape

        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            precision * 5, 0.01
        )

        img_copy = img.reshape(-1, c)
        _, label, center = cv2.kmeans(
            img_copy, colors, None,
            criteria, 1, cv2.KMEANS_PP_CENTERS
        )

        result = center[label.flatten()].reshape(*img.shape)
        tensor = torch.from_numpy(result).unsqueeze(0)
        return (tensor,)
