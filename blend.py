import numpy as np
import torch

class Blend:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "blend_factor": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "blend_mode": (["normal", "multiply", "screen", "overlay", "soft_light"],),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "blend_images"

    CATEGORY = "postprocessing"

    def blend_images(self, image1: torch.Tensor, image2: torch.Tensor, blend_factor: float, blend_mode: str):
        batch_size, height, width, _ = image1.shape
        result = torch.zeros_like(image1)

        for b in range(batch_size):
            img1 = image1[b].numpy()
            img2 = image2[b].numpy()

            blended_image = self.blend_mode(img1, img2, blend_mode)
            blended_image = img1 * (1 - blend_factor) + blended_image * blend_factor
            blended_image = np.clip(blended_image, 0, 1)

            tensor = torch.from_numpy(blended_image).unsqueeze(0)
            result[b] = tensor

        return (result,)

    def blend_mode(self, img1, img2, mode):
        if mode == "normal":
            return img2
        elif mode == "multiply":
            return img1 * img2
        elif mode == "screen":
            return 1 - (1 - img1) * (1 - img2)
        elif mode == "overlay":
            return np.where(img1 <= 0.5, 2 * img1 * img2, 1 - 2 * (1 - img1) * (1 - img2))
        elif mode == "soft_light":
            return np.where(img2 <= 0.5, img1 - (1 - 2 * img2) * img1 * (1 - img1), img1 + (2 * img2 - 1) * (self.g(img1) - img1))
        else:
            raise ValueError(f"Unsupported blend mode: {mode}")

    def g(self, x):
        return np.where(x <= 0.25, ((16 * x - 12) * x + 4) * x, np.sqrt(x))

NODE_CLASS_MAPPINGS = {
    "Blend": Blend,
}
