import torch
import torch.nn.functional as F

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

    CATEGORY = "postprocessing/Blends"

    def blend_images(self, image1: torch.Tensor, image2: torch.Tensor, blend_factor: float, blend_mode: str):
        if image1.shape != image2.shape:
            image2 = self.crop_and_resize(image2, image1.shape)

        blended_image = self.blend_mode(image1, image2, blend_mode)
        blended_image = image1 * (1 - blend_factor) + blended_image * blend_factor
        blended_image = torch.clamp(blended_image, 0, 1)
        return (blended_image,)

    def blend_mode(self, img1, img2, mode):
        if mode == "normal":
            return img2
        elif mode == "multiply":
            return img1 * img2
        elif mode == "screen":
            return 1 - (1 - img1) * (1 - img2)
        elif mode == "overlay":
            return torch.where(img1 <= 0.5, 2 * img1 * img2, 1 - 2 * (1 - img1) * (1 - img2))
        elif mode == "soft_light":
            return torch.where(img2 <= 0.5, img1 - (1 - 2 * img2) * img1 * (1 - img1), img1 + (2 * img2 - 1) * (self.g(img1) - img1))
        else:
            raise ValueError(f"Unsupported blend mode: {mode}")

    def g(self, x):
        return torch.where(x <= 0.25, ((16 * x - 12) * x + 4) * x, torch.sqrt(x))

    def crop_and_resize(self, img: torch.Tensor, target_shape: tuple):
        batch_size, img_h, img_w, img_c = img.shape
        _, target_h, target_w, _ = target_shape
        img_aspect_ratio = img_w / img_h
        target_aspect_ratio = target_w / target_h

        # Crop center of the image to the target aspect ratio
        if img_aspect_ratio > target_aspect_ratio:
            new_width = int(img_h * target_aspect_ratio)
            left = (img_w - new_width) // 2
            img = img[:, :, left:left + new_width, :]
        else:
            new_height = int(img_w / target_aspect_ratio)
            top = (img_h - new_height) // 2
            img = img[:, top:top + new_height, :, :]

        # Resize to target size
        img = img.permute(0, 3, 1, 2) # Torch wants (B, C, H, W) we use (B, H, W, C)
        img = F.interpolate(img, size=(target_h, target_w), mode='bilinear', align_corners=False)
        img = img.permute(0, 2, 3, 1)

        return img

NODE_CLASS_MAPPINGS = {
    "Blend": Blend,
}
