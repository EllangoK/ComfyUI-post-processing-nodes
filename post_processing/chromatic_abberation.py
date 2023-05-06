import torch

class ChromaticAberration:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "red_shift": ("INT", {
                    "default": 0,
                    "min": -20,
                    "max": 20,
                    "step": 1
                }),
                "red_direction": (["horizontal", "vertical"],),
                "green_shift": ("INT", {
                    "default": 0,
                    "min": -20,
                    "max": 20,
                    "step": 1
                }),
                "green_direction": (["horizontal", "vertical"],),
                "blue_shift": ("INT", {
                    "default": 0,
                    "min": -20,
                    "max": 20,
                    "step": 1
                }),
                "blue_direction": (["horizontal", "vertical"],),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "chromatic_aberration"

    CATEGORY = "postprocessing/Effects"

    def chromatic_aberration(self, image: torch.Tensor, red_shift: int, green_shift: int, blue_shift: int, red_direction: str, green_direction: str, blue_direction: str):
        def get_shift(direction, shift):
            shift = -shift if direction == 'vertical' else shift # invert vertical shift as otherwise positive actually shifts down
            return (shift, 0) if direction == 'vertical' else (0, shift)

        x = image.permute(0, 3, 1, 2)
        shifts = [get_shift(direction, shift) for direction, shift in zip([red_direction, green_direction, blue_direction], [red_shift, green_shift, blue_shift])]
        channels = [torch.roll(x[:, i, :, :], shifts=shifts[i], dims=(1, 2)) for i in range(3)]

        output = torch.stack(channels, dim=1)
        output = output.permute(0, 2, 3, 1)

        return (output,)

NODE_CLASS_MAPPINGS = {
    "ChromaticAberration": ChromaticAberration
}
