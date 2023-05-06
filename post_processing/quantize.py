import torch
from PIL import Image
import numpy as np

class Quantize:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "colors": ("INT", {
                    "default": 256,
                    "min": 1,
                    "max": 256,
                    "step": 1
                }),
                "dither": (["none", "floyd-steinberg"],),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "quantize"

    CATEGORY = "postprocessing/Color Adjustments"

    def quantize(self, image: torch.Tensor, colors: int = 256, dither: str = "FLOYDSTEINBERG"):
        batch_size, height, width, _ = image.shape
        result = torch.zeros_like(image)

        dither_option = Image.Dither.FLOYDSTEINBERG if dither == "floyd-steinberg" else Image.Dither.NONE

        for b in range(batch_size):
            tensor_image = image[b]
            img = (tensor_image * 255).to(torch.uint8).numpy()
            pil_image = Image.fromarray(img, mode='RGB')

            palette = pil_image.quantize(colors=colors) # Required as described in https://github.com/python-pillow/Pillow/issues/5836
            quantized_image = pil_image.quantize(colors=colors, palette=palette, dither=dither_option)

            quantized_array = torch.tensor(np.array(quantized_image.convert("RGB"))).float() / 255
            result[b] = quantized_array

        return (result,)

NODE_CLASS_MAPPINGS = {
    "Quantize": Quantize,
}