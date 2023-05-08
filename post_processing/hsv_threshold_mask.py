import cv2
import torch
import numpy as np

class HSVThresholdMask:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "low_threshold": ("FLOAT", {
                    "default": 0.2,
                    "min": 0,
                    "max": 1,
                    "step": 0.1
                }),
                "high_threshold": ("FLOAT", {
                    "default": 0.7,
                    "min": 0,
                    "max": 1,
                    "step": 0.1
                }),
                "hsv_channel": (["hue", "saturation", "value"],),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "hsv_threshold"

    CATEGORY = "postprocessing/Masks"

    def hsv_threshold(self, image: torch.Tensor, low_threshold: float, high_threshold: float, hsv_channel: str):
        batch_size, height, width, _ = image.shape
        result = torch.zeros(batch_size, height, width)

        if hsv_channel == "hue":
            channel = 0
            low_threshold, high_threshold = int(low_threshold * 180), int(high_threshold * 180)
        elif hsv_channel == "saturation":
            channel = 1
            low_threshold, high_threshold = int(low_threshold * 255), int(high_threshold * 255)
        elif hsv_channel == "value":
            channel = 2
            low_threshold, high_threshold = int(low_threshold * 255), int(high_threshold * 255)

        for b in range(batch_size):
            tensor_image = (image[b].numpy().copy() * 255).astype(np.uint8)
            hsv_image = cv2.cvtColor(tensor_image, cv2.COLOR_RGB2HSV)

            mask = cv2.inRange(hsv_image[:, :, channel], low_threshold, high_threshold)
            tensor = torch.from_numpy(mask).float() / 255.
            result[b] = tensor

        return (result,)

NODE_CLASS_MAPPINGS = {
    "HSVThresholdMask": HSVThresholdMask,
}
