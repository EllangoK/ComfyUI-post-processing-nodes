import torch

class Dissolve:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "dissolve_factor": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "dissolve_images"

    CATEGORY = "postprocessing/Blends"

    def dissolve_images(self, image1: torch.Tensor, image2: torch.Tensor, dissolve_factor: float):
        dither_pattern = torch.rand_like(image1)
        mask = (dither_pattern < dissolve_factor).float()

        dissolved_image = image1 * mask + image2 * (1 - mask)
        dissolved_image = torch.clamp(dissolved_image, 0, 1)
        return (dissolved_image,)

NODE_CLASS_MAPPINGS = {
    "Dissolve": Dissolve,
}
