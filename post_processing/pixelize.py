import torch
import torch.nn.functional as F

class Pixelize:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "pixel_size": ("INT", {
                    "default": 8,
                    "min": 2,
                    "max": 128,
                    "step": 1
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_pixelize"

    CATEGORY = "postprocessing/Effects"

    def apply_pixelize(self, image: torch.Tensor, pixel_size: int):
        pixelized_image = self.pixelize_image(image, pixel_size)
        pixelized_image = torch.clamp(pixelized_image, 0, 1)
        return (pixelized_image,)

    def pixelize_image(self, image: torch.Tensor, pixel_size: int):
        batch_size, height, width, channels = image.shape
        new_height = height // pixel_size
        new_width = width // pixel_size

        image = image.permute(0, 3, 1, 2)
        image = F.avg_pool2d(image, kernel_size=pixel_size, stride=pixel_size)
        image = F.interpolate(image, size=(height, width), mode='nearest')
        image = image.permute(0, 2, 3, 1)

        return image

NODE_CLASS_MAPPINGS = {
    "Pixelize": Pixelize,
}
