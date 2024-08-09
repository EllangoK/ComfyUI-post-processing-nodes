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
                "vignette": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.01
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

        radius = radius / torch.amax(radius, dim=(0, 1), keepdim=True)
        opacity = torch.tensor(vignette, device=image.device)
        opacity = torch.clamp(opacity, 0.0, 1.0)
        vignette = 1 - radius.unsqueeze(0).unsqueeze(-1) * opacity

        vignette_image = torch.clamp(image * vignette, 0, 1)

        return (vignette_image,)

NODE_CLASS_MAPPINGS = {
    "Vignette": Vignette,
}
