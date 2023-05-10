from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch

class AsciiArt:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "char_size": ("INT", {
                    "default": 12,
                    "min": 0,
                    "max": 64,
                    "step": 2,
                }),
                "font_size": ("INT", {
                    "default": 12,
                    "min": 0,
                    "max": 64,
                    "step": 2,
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_ascii_art_effect"

    CATEGORY = "postprocessing/Effects"

    def apply_ascii_art_effect(self, image: torch.Tensor, char_size: int, font_size: int):
        batch_size, height, width, channels = image.shape
        result = torch.zeros_like(image)

        for b in range(batch_size):
            img_b = image[b] * 255.0
            img_b = Image.fromarray(img_b.numpy().astype('uint8'), 'RGB')
            result_b = ascii_art_effect(img_b, char_size, font_size)
            result_b = torch.tensor(np.array(result_b)) / 255.0
            result[b] = result_b

        return (result,)


def ascii_art_effect(image: torch.Tensor, char_size: int, font_size: int):
    chars = " .'`^\",:;I1!i><-+_-?][}{1)(|\/tfjrxnuvczXYUCLQ0OZmwqpbdkhao*#MW&8%B@$"
    small_image = image.resize((image.size[0] // char_size, image.size[1] // char_size), Image.Resampling.NEAREST)

    def get_char(value):
        return chars[value * len(chars) // 256]

    ascii_image = Image.new('RGB', image.size, (0, 0, 0))
    font = ImageFont.truetype("arial.ttf", font_size)
    draw_image = ImageDraw.Draw(ascii_image)

    for i in range(small_image.height):
        for j in range(small_image.width):
            r, g, b = small_image.getpixel((j, i))
            k = (r + g + b) // 3
            draw_image.text(
                (j * char_size, i * char_size),
                get_char(k),
                font=font,
                fill=(r, g, b)
            )

    return ascii_image

NODE_CLASS_MAPPINGS = {
    "AsciiArt": AsciiArt,
}