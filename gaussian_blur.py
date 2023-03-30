import cv2
import torch

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
        tensor_image = image.numpy()[0]
        blurred = cv2.GaussianBlur(tensor_image, (kernel_size, kernel_size), sigma)
        tensor = torch.from_numpy(blurred).unsqueeze(0)
        return (tensor,)

NODE_CLASS_MAPPINGS = {
    "GaussianBlur": GaussianBlur
}