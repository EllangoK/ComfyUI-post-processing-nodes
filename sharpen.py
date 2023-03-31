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
        result = torch.zeros_like(image)

        kernel = torch.ones((channels, 1, kernel_size, kernel_size), dtype=torch.float32) * -1
        center = kernel_size // 2
        kernel[:, 0, center, center] = kernel_size**2
        kernel *= alpha

        for b in range(batch_size):
            tensor_image = image[b].permute(2, 0, 1).unsqueeze(0)

            sharpened = F.conv2d(tensor_image, kernel, padding=center, groups=channels)
            sharpened = sharpened.squeeze(0).permute(1, 2, 0)

            tensor = torch.clamp(sharpened, 0, 1)
            result[b] = tensor

        return (result,)

NODE_CLASS_MAPPINGS = {
    "Sharpen": Sharpen
}
