import torch

class Sepia:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.1
                }),
                "mode": (["sepia", "blue-pia", "green-pia"],),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "sepia"

    CATEGORY = "postprocessing/Color Adjustments"

    def sepia(self, image: torch.Tensor, strength: float, mode: str = "sepia"):
        if strength == 0:
            return (image,)

        sepia_weights = torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 1, 1, 3).to(image.device)

        if mode == "sepia":
            sepia_filter = torch.tensor([1.0, 0.8, 0.6]).view(1, 1, 1, 3).to(image.device)
        elif mode == "blue-pia":
            sepia_filter = torch.tensor([0.6, 0.8, 1.0]).view(1, 1, 1, 3).to(image.device)
        elif mode == "green-pia":
            sepia_filter = torch.tensor([0.6, 1.0, 0.6]).view(1, 1, 1, 3).to(image.device)

        grayscale = torch.sum(image * sepia_weights, dim=-1, keepdim=True)
        sepia = grayscale * sepia_filter

        result = sepia * strength + image * (1 - strength)
        return (result,)

NODE_CLASS_MAPPINGS = {
    "Sepia": Sepia
}
