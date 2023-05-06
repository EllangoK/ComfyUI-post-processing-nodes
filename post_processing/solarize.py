import torch

class Solarize:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "solarize_image"

    CATEGORY = "postprocessing/Color Adjustments"

    def solarize_image(self, image: torch.Tensor, threshold: float):
        solarized_image = torch.where(image > threshold, 1 - image, image)
        solarized_image = torch.clamp(solarized_image, 0, 1)
        return (solarized_image,)

NODE_CLASS_MAPPINGS = {
    "Solarize": Solarize,
}
