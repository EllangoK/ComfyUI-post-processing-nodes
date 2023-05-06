import torch
import torch.nn.functional as F

class Glow:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "intensity": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.01
                }),
                "blur_radius": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 50,
                    "step": 1
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_glow"

    CATEGORY = "postprocessing/Effects"

    def apply_glow(self, image: torch.Tensor, intensity: float, blur_radius: int):
        blurred_image = self.gaussian_blur(image, 2 * blur_radius + 1)
        glowing_image = self.add_glow(image, blurred_image, intensity)
        glowing_image = torch.clamp(glowing_image, 0, 1)
        return (glowing_image,)

    def gaussian_blur(self, image: torch.Tensor, kernel_size: int):
        batch_size, height, width, channels = image.shape

        sigma = (kernel_size - 1) / 6
        kernel = gaussian_kernel(kernel_size, sigma).repeat(channels, 1, 1).unsqueeze(1)

        image = image.permute(0, 3, 1, 2) # Torch wants (B, C, H, W) we use (B, H, W, C)
        blurred = F.conv2d(image, kernel, padding=kernel_size // 2, groups=channels)
        blurred = blurred.permute(0, 2, 3, 1)

        return blurred

    def add_glow(self, img, blurred_img, intensity):
        return img + blurred_img * intensity

NODE_CLASS_MAPPINGS = {
    "Glow": Glow,
}

def gaussian_kernel(kernel_size: int, sigma: float):
    x, y = torch.meshgrid(torch.linspace(-1, 1, kernel_size), torch.linspace(-1, 1, kernel_size), indexing="ij")
    d = torch.sqrt(x * x + y * y)
    g = torch.exp(-(d * d) / (2.0 * sigma * sigma))
    return g / g.sum()
