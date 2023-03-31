import torch
import torch.nn.functional as F


class Sharpen:
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
                "alpha": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "sharpen"

    CATEGORY = "postprocessing"

    def sharpen(self, image: torch.Tensor, kernel_size: int, alpha: float):
        batch_size, height, width, channels = image.shape

        kernel = torch.ones((kernel_size, kernel_size), dtype=torch.float32) * -1
        center = kernel_size // 2
        kernel[center, center] = kernel_size**2
        kernel *= alpha
        kernel = kernel.repeat(channels, 1, 1).unsqueeze(1)

        tensor_image = image.permute(0, 3, 1, 2) # Torch wants (B, C, H, W) we use (B, H, W, C)
        sharpened = F.conv2d(tensor_image, kernel, padding=center, groups=channels)
        sharpened = sharpened.permute(0, 2, 3, 1)

        result = torch.clamp(sharpened, 0, 1)

        return (result,)

NODE_CLASS_MAPPINGS = {
    "Sharpen": Sharpen
}
