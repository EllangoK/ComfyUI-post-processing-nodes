import torch

class Dither:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "bits": ("INT", {
                    "default": 4, 
                    "min": 0,
                    "max": 8,
                    "step": 1
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "dither"

    CATEGORY = "postprocessing"

    def dither(self, image, bits):
        tensor_img = image[0]
        height, width, _ = tensor_img.shape
        out = tensor_img.clone()
        levels = 2 ** bits - 1

        for y in range(height):
            for x in range(width):
                old_pixel = out[y, x].clone()
                new_pixel = torch.round(old_pixel * levels) / levels
                out[y, x] = new_pixel

                error = old_pixel - new_pixel

                if x + 1 < width:
                    out[y, x + 1] += error * (7 / 16)
                if x - 1 >= 0 and y + 1 < height:
                    out[y + 1, x - 1] += error * 3/16
                if y + 1 < height:
                    out[y + 1, x] += error * 5/16
                if x + 1 < width and y + 1 < height:
                    out[y + 1, x + 1] += error * 1/16

        out = torch.clamp(out, 0, 1).unsqueeze(0)

        return (out,)


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "Dither": Dither
}
