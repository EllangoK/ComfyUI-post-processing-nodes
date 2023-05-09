import torch
import numpy as np

class SineWave:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "amplitude": ("FLOAT", {
                    "default": 10,
                    "min": 0,
                    "max": 150,
                    "step": 5
                }),
                "frequency": ("FLOAT", {
                    "default": 5,
                    "min": 0,
                    "max": 20,
                    "step": 1
                }),
                "direction": (["horizontal", "vertical"],),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_sine_wave"

    CATEGORY = "postprocessing/Effects"

    def apply_sine_wave(self, image: torch.Tensor, amplitude: float, frequency: float, direction: str):
        batch_size, height, width, channels = image.shape
        result = torch.zeros_like(image)

        for b in range(batch_size):
            tensor_image = image[b]
            result[b] = self.sine_wave_effect(tensor_image, amplitude, frequency, direction)

        return (result,)

    def sine_wave_effect(self, image: torch.Tensor, amplitude: float, frequency: float, direction: str):
        height, width, _ = image.shape
        shifted_image = torch.zeros_like(image)

        for channel in range(3):
            if direction == "horizontal":
                for i in range(height):
                    offset = int(amplitude * np.sin(2 * torch.pi * i * frequency / height))
                    shifted_image[i, :, channel] = torch.roll(image[i, :, channel], offset)
            elif direction == "vertical":
                for j in range(width):
                    offset = int(amplitude * np.sin(2 * torch.pi * j * frequency / width))
                    shifted_image[:, j, channel] = torch.roll(image[:, j, channel], offset)

        return shifted_image

NODE_CLASS_MAPPINGS = {
    "SineWave": SineWave,
}
