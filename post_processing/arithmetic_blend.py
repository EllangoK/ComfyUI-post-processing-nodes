import torch

class ArithmeticBlend:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "blend_mode": (["add", "subtract", "difference"],),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "arithmetic_blend_images"

    CATEGORY = "postprocessing/Blends"

    def arithmetic_blend_images(self, image1: torch.Tensor, image2: torch.Tensor, blend_mode: str):
        if blend_mode == "add":
            blended_image = self.add(image1, image2)
        elif blend_mode == "subtract":
            blended_image = self.subtract(image1, image2)
        elif blend_mode == "difference":
            blended_image = self.difference(image1, image2)
        else:
            raise ValueError(f"Unsupported arithmetic blend mode: {blend_mode}")

        blended_image = torch.clamp(blended_image, 0, 1)
        return (blended_image,)

    def add(self, img1, img2):
        return img1 + img2

    def subtract(self, img1, img2):
        return img1 - img2

    def difference(self, img1, img2):
        return torch.abs(img1 - img2)

NODE_CLASS_MAPPINGS = {
    "ArithmeticBlend": ArithmeticBlend,
}
