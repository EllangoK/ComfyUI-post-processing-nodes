import numpy as np
import torch

class Vignette:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "a": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 1.0
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_vignette"

    CATEGORY = "postprocessing/Effects"

    def apply_vignette(self, image: torch.Tensor, vignette: float):
        if vignette == 0:
            return (image,)
        height, width, _ = image.shape[-3:]
        x = torch.linspace(-1, 1, width, device=image.device)
        y = torch.linspace(-1, 1, height, device=image.device)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        radius = torch.sqrt(X ** 2 + Y ** 2)

        # Map vignette strength from 0-10 to 1.800-0.800
        mapped_vignette_strength = 1.8 - (vignette - 1) * 0.1
        vignette = 1 - torch.clamp(radius / mapped_vignette_strength, 0, 1)
        vignette = vignette[..., None]

        vignette_image = torch.clamp(image * vignette, 0, 1)

        return (vignette_image,)

NODE_CLASS_MAPPINGS = {
    "Vignette": Vignette,
}
