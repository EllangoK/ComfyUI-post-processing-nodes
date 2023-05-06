import torch
import torch.nn.functional as F

class PencilSketch:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "blur_radius": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 31,
                    "step": 1
                }),
                "sharpen_alpha": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_sketch"

    CATEGORY = "postprocessing/Effects"

    def apply_sketch(self, image: torch.Tensor, blur_radius: int = 5, sharpen_alpha: float = 1):
        image = image.permute(0, 3, 1, 2)  # Torch wants (B, C, H, W) we use (B, H, W, C)

        grayscale = image.mean(dim=1, keepdim=True)
        grayscale = grayscale.repeat(1, 3, 1, 1)
        inverted = 1 - grayscale

        blur_sigma = blur_radius / 3
        blurred = self.gaussian_blur(inverted, blur_radius, blur_sigma)

        final_image = self.dodge(blurred, grayscale)

        if sharpen_alpha != 0.0:
            final_image = self.sharpen(final_image, 1, sharpen_alpha)

        final_image = final_image.permute(0, 2, 3, 1)  # Back to (B, H, W, C)

        return (final_image,)

    def dodge(self, front: torch.Tensor, back: torch.Tensor) -> torch.Tensor:
        result = back / (1 - front + 1e-7)
        result = torch.clamp(result, 0, 1)
        return result

    def gaussian_blur(self, image: torch.Tensor, blur_radius: int, sigma: float):
        if blur_radius == 0:
            return image

        batch_size, channels, height, width = image.shape

        kernel_size = blur_radius * 2 + 1
        kernel = gaussian_kernel(kernel_size, sigma).repeat(channels, 1, 1).unsqueeze(1)

        blurred = F.conv2d(image, kernel, padding=kernel_size // 2, groups=channels)

        return blurred

    def sharpen(self, image: torch.Tensor, blur_radius: int, alpha: float):
        if blur_radius == 0:
            return image

        batch_size, channels, height, width = image.shape

        kernel_size = blur_radius * 2 + 1
        kernel = torch.ones((kernel_size, kernel_size), dtype=torch.float32) * -1
        center = kernel_size // 2
        kernel[center, center] = kernel_size**2
        kernel *= alpha
        kernel = kernel.repeat(channels, 1, 1).unsqueeze(1)

        sharpened = F.conv2d(image, kernel, padding=center, groups=channels)

        result = torch.clamp(sharpened, 0, 1)

        return result

NODE_CLASS_MAPPINGS = {
    "PencilSketch": PencilSketch,
}

def gaussian_kernel(kernel_size: int, sigma: float):
    x, y = torch.meshgrid(torch.linspace(-1, 1, kernel_size), torch.linspace(-1, 1, kernel_size), indexing="ij")
    d = torch.sqrt(x * x + y * y)
    g = torch.exp(-(d * d) / (2.0 * sigma * sigma))
    return g / g.sum()
