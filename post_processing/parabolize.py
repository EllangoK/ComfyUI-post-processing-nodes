import torch

class Parabolize:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "coeff": ("FLOAT", {
                    "default": 1.0,
                    "min": -10.0,
                    "max": 10.0,
                    "step": 0.1
                }),
                "vertex_x": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1
                }),
                "vertex_y": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "parabolize_image"

    CATEGORY = "postprocessing/Color Adjustments"

    def parabolize_image(self, image: torch.Tensor, coeff: float, vertex_x: float, vertex_y: float):
        parabolized_image = coeff * torch.pow(image - vertex_x, 2) + vertex_y
        parabolized_image = torch.clamp(parabolized_image, 0, 1)
        return (parabolized_image,)

NODE_CLASS_MAPPINGS = {
    "Parabolize": Parabolize,
}
