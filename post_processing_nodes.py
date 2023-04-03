import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image, ImageEnhance
from PIL import Image


class ArithmeticBlend:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "blend_mode": (["add", "subtract", "difference"],),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "arithmetic_blend_images"

    CATEGORY = "postprocessing"

    def arithmetic_blend_images(self, image1: torch.Tensor, image2: torch.Tensor, blend_mode: str):
        if blend_mode == "add":
            blended_image = self.add(image1, image2)
        elif blend_mode == "subtract":
            blended_image = self.subtract(image1, image2)
        elif blend_mode == "difference":
            blended_image = self.difference(image1, image2)
        else:
            raise ValueError(f"Unsupported arithmetic blend mode: {blend_mode}")

        blended_image = torch.clamp(blended_image, 0, 1)
        return (blended_image,)

    def add(self, img1, img2):
        return img1 + img2

    def subtract(self, img1, img2):
        return img1 - img2

    def difference(self, img1, img2):
        return torch.abs(img1 - img2)

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

    CATEGORY = "postprocessing"

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

class Blur:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "blur_radius": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 15,
                    "step": 1
                }),
                "sigma": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "blur"

    CATEGORY = "postprocessing"

    def blur(self, image: torch.Tensor, blur_radius: int, sigma: float):
        if blur_radius == 0:
            return (image,)

        batch_size, height, width, channels = image.shape

        kernel_size = blur_radius * 2 + 1
        kernel = gaussian_kernel(kernel_size, sigma).repeat(channels, 1, 1).unsqueeze(1)

        image = image.permute(0, 3, 1, 2) # Torch wants (B, C, H, W) we use (B, H, W, C)
        blurred = F.conv2d(image, kernel, padding=kernel_size // 2, groups=channels)
        blurred = blurred.permute(0, 2, 3, 1)

        return (blurred,)

class CannyEdgeDetection:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "lower_threshold": ("INT", {
                    "default": 100,
                    "min": 0,
                    "max": 500,
                    "step": 10
                }),
                "upper_threshold": ("INT", {
                    "default": 200,
                    "min": 0,
                    "max": 500,
                    "step": 10
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "canny"

    CATEGORY = "postprocessing"

    def canny(self, image: torch.Tensor, lower_threshold: int, upper_threshold: int):
        batch_size, height, width, _ = image.shape
        result = torch.zeros(batch_size, height, width)

        for b in range(batch_size):
            tensor_image = image[b].numpy().copy()
            gray_image = (cv2.cvtColor(tensor_image, cv2.COLOR_RGB2GRAY) * 255).astype(np.uint8)
            canny = cv2.Canny(gray_image, lower_threshold, upper_threshold)
            tensor = torch.from_numpy(canny)
            result[b] = tensor

        return (result,)

class ColorCorrect:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "temperature": ("FLOAT", {
                    "default": 0,
                    "min": -100,
                    "max": 100,
                    "step": 5
                }),
                "hue": ("FLOAT", {
                    "default": 0,
                    "min": -90,
                    "max": 90,
                    "step": 5
                }),
                "brightness": ("FLOAT", {
                    "default": 0,
                    "min": -100,
                    "max": 100,
                    "step": 5
                }),
                "contrast": ("FLOAT", {
                    "default": 0,
                    "min": -100,
                    "max": 100,
                    "step": 5
                }),
                "saturation": ("FLOAT", {
                    "default": 0,
                    "min": -100,
                    "max": 100,
                    "step": 5
                }),
                "gamma": ("FLOAT", {
                    "default": 1,
                    "min": 0.2,
                    "max": 2.2,
                    "step": 0.1
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "color_correct"

    CATEGORY = "postprocessing"

    def color_correct(self, image: torch.Tensor, temperature: float, hue: float, brightness: float, contrast: float, saturation: float, gamma: float):
        batch_size, height, width, _ = image.shape
        result = torch.zeros_like(image)

        brightness /= 100
        contrast /= 100
        saturation /= 100
        temperature /= 100

        brightness = 1 + brightness
        contrast = 1 + contrast
        saturation = 1 + saturation

        for b in range(batch_size):
            tensor_image = image[b].numpy()

            modified_image = Image.fromarray((tensor_image * 255).astype(np.uint8))

            # brightness
            modified_image = ImageEnhance.Brightness(modified_image).enhance(brightness)

            # contrast
            modified_image = ImageEnhance.Contrast(modified_image).enhance(contrast)
            modified_image = np.array(modified_image).astype(np.float32)

            # temperature
            if temperature > 0:
                modified_image[:, :, 0] *= 1 + temperature
                modified_image[:, :, 1] *= 1 + temperature * 0.4
            elif temperature < 0:
                modified_image[:, :, 2] *= 1 - temperature
            modified_image = np.clip(modified_image, 0, 255)/255

            # gamma
            modified_image = np.clip(np.power(modified_image, gamma), 0, 1)

            # saturation
            hls_img = cv2.cvtColor(modified_image, cv2.COLOR_RGB2HLS)
            hls_img[:, :, 2] = np.clip(saturation*hls_img[:, :, 2], 0, 1)
            modified_image = cv2.cvtColor(hls_img, cv2.COLOR_HLS2RGB) * 255

            # hue
            hsv_img = cv2.cvtColor(modified_image, cv2.COLOR_RGB2HSV)
            hsv_img[:, :, 0] = (hsv_img[:, :, 0] + hue) % 360
            modified_image = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)

            modified_image = modified_image.astype(np.uint8)
            modified_image = modified_image / 255
            modified_image = torch.from_numpy(modified_image).unsqueeze(0)
            result[b] = modified_image

        return (result, )

class Dissolve:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "dissolve_factor": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "dissolve_images"

    CATEGORY = "postprocessing"

    def dissolve_images(self, image1: torch.Tensor, image2: torch.Tensor, dissolve_factor: float):
        dither_pattern = torch.rand_like(image1)
        mask = (dither_pattern < dissolve_factor).float()

        dissolved_image = image1 * mask + image2 * (1 - mask)
        dissolved_image = torch.clamp(dissolved_image, 0, 1)
        return (dissolved_image,)

class DodgeAndBurn:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("IMAGE",),
                "intensity": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "mode": (["dodge", "burn", "dodge_and_burn", "burn_and_dodge", "color_dodge", "color_burn", "linear_dodge", "linear_burn"],),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "dodge_and_burn"

    CATEGORY = "postprocessing"

    def dodge_and_burn(self, image: torch.Tensor, mask: torch.Tensor, intensity: float, mode: str):
        if mode in ["dodge", "color_dodge", "linear_dodge"]:
            dodged_image = self.dodge(image, mask, intensity, mode)
            return (dodged_image,)
        elif mode in ["burn", "color_burn", "linear_burn"]:
            burned_image = self.burn(image, mask, intensity, mode)
            return (burned_image,)
        elif mode == "dodge_and_burn":
            dodged_image = self.dodge(image, mask, intensity, "dodge")
            burned_image = self.burn(dodged_image, mask, intensity, "burn")
            return (burned_image,)
        elif mode == "burn_and_dodge":
            burned_image = self.burn(image, mask, intensity, "burn")
            dodged_image = self.dodge(burned_image, mask, intensity, "dodge")
            return (dodged_image,)
        else:
            raise ValueError(f"Unsupported dodge and burn mode: {mode}")

    def dodge(self, img, mask, intensity, mode):
        if mode == "dodge":
            return img / (1 - mask * intensity + 1e-7)
        elif mode == "color_dodge":
            return torch.where(mask < 1, img / (1 - mask * intensity), img)
        elif mode == "linear_dodge":
            return torch.clamp(img + mask * intensity, 0, 1)
        else:
            raise ValueError(f"Unsupported dodge mode: {mode}")

    def burn(self, img, mask, intensity, mode):
        if mode == "burn":
            return 1 - (1 - img) / (mask * intensity + 1e-7)
        elif mode == "color_burn":
            return torch.where(mask > 0, 1 - (1 - img) / (mask * intensity), img)
        elif mode == "linear_burn":
            return torch.clamp(img - mask * intensity, 0, 1)
        else:
            raise ValueError(f"Unsupported burn mode: {mode}")

class FilmGrain:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "intensity": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "scale": ("FLOAT", {
                    "default": 10,
                    "min": 1,
                    "max": 100,
                    "step": 1
                }),
                "temperature": ("FLOAT", {
                    "default": 0.0,
                    "min": -100,
                    "max": 100,
                    "step": 1
                }),
                "vignette": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 1.0
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "film_grain"

    CATEGORY = "postprocessing"

    def film_grain(self, image: torch.Tensor, intensity: float, scale: float, temperature: float, vignette: float):
        batch_size, height, width, _ = image.shape
        result = torch.zeros_like(image)

        for b in range(batch_size):
            tensor_image = image[b].numpy()

            # Generate Perlin noise with shape (height, width) and scale
            noise = self.generate_perlin_noise((height, width), scale)
            noise = (noise - np.min(noise)) / (np.max(noise) - np.min(noise))

            # Apply grain intensity
            noise = (noise * 2 - 1) * intensity

            # Blend the noise with the image
            grain_image = np.clip(tensor_image + noise[:, :, np.newaxis], 0, 1)

            # Apply temperature
            grain_image = self.apply_temperature(grain_image, temperature)

            # Apply vignette
            grain_image = self.apply_vignette(grain_image, vignette)

            tensor = torch.from_numpy(grain_image).unsqueeze(0)
            result[b] = tensor

        return (result,)

    def generate_perlin_noise(self, shape, scale, octaves=4, persistence=0.5, lacunarity=2):
        def smoothstep(t):
            return t * t * (3.0 - 2.0 * t)

        def lerp(t, a, b):
            return a + t * (b - a)

        def gradient(h, x, y):
            vectors = np.array([[1, 1], [-1, 1], [1, -1], [-1, -1]])
            g = vectors[h % 4]
            return g[:, :, 0] * x + g[:, :, 1] * y

        height, width = shape
        noise = np.zeros(shape)

        for octave in range(octaves):
            octave_scale = scale * lacunarity ** octave
            x = np.linspace(0, 1, width, endpoint=False)
            y = np.linspace(0, 1, height, endpoint=False)
            X, Y = np.meshgrid(x, y)
            X, Y = X * octave_scale, Y * octave_scale

            xi = X.astype(int)
            yi = Y.astype(int)

            xf = X - xi
            yf = Y - yi

            u = smoothstep(xf)
            v = smoothstep(yf)

            n00 = gradient(np.random.randint(0, 4, (height, width)), xf, yf)
            n01 = gradient(np.random.randint(0, 4, (height, width)), xf, yf - 1)
            n10 = gradient(np.random.randint(0, 4, (height, width)), xf - 1, yf)
            n11 = gradient(np.random.randint(0, 4, (height, width)), xf - 1, yf - 1)

            x1 = lerp(u, n00, n10)
            x2 = lerp(u, n01, n11)
            y1 = lerp(v, x1, x2)

            noise += y1 * persistence ** octave

        return noise / (1 - persistence ** octaves)

    def apply_temperature(self, image, temperature):
        if temperature == 0:
            return image

        temperature /= 100

        new_image = image.copy()

        if temperature > 0:
            new_image[:, :, 0] *= 1 + temperature
            new_image[:, :, 1] *= 1 + temperature * 0.4
        else:
            new_image[:, :, 2] *= 1 - temperature

        return np.clip(new_image, 0, 1)

    def apply_vignette(self, image, vignette_strength):
        if vignette_strength == 0:
            return image

        height, width, _ = image.shape
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)
        X, Y = np.meshgrid(x, y)
        radius = np.sqrt(X ** 2 + Y ** 2)

        # Map vignette strength from 0-10 to 1.800-0.800
        mapped_vignette_strength = 1.8 - (vignette_strength - 1) * 0.1
        vignette = 1 - np.clip(radius / mapped_vignette_strength, 0, 1)

        return np.clip(image * vignette[..., np.newaxis], 0, 1)

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

    CATEGORY = "postprocessing"

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

    CATEGORY = "postprocessing"

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

class PixelSort:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("IMAGE",),
                "direction": (["horizontal", "vertical"],),
                "span_limit": ("INT", {
                    "default": None,
                    "min": 0,
                    "max": 100,
                    "step": 5
                }),
                "sort_by": (["hue", "saturation", "value"],),
                "order": (["forward", "backward"],),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "sort_pixels"

    CATEGORY = "postprocessing"

    def sort_pixels(self, image: torch.Tensor, mask: torch.Tensor, direction: str, span_limit: int, sort_by: str, order: str):
        horizontal_sort = direction == "horizontal"
        reverse_sorting = order == "backward"
        sort_by = sort_by[0].upper()
        span_limit = span_limit if span_limit > 0 else None

        batch_size = image.shape[0]
        result = torch.zeros_like(image)

        for b in range(batch_size):
            tensor_img = image[b].numpy()
            tensor_mask = mask[b].numpy()
            sorted_image = pixel_sort(tensor_img, tensor_mask, horizontal_sort, span_limit, sort_by, reverse_sorting)
            result[b] = torch.from_numpy(sorted_image)

        return (result,)

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

    CATEGORY = "postprocessing"

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

    CATEGORY = "postprocessing"

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

class Sharpen:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "sharpen_radius": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 15,
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

    def sharpen(self, image: torch.Tensor, blur_radius: int, alpha: float):
        if blur_radius == 0:
            return (image,)

        batch_size, height, width, channels = image.shape

        kernel_size = blur_radius * 2 + 1
        kernel = torch.ones((kernel_size, kernel_size), dtype=torch.float32) * -1
        center = kernel_size // 2
        kernel[center, center] = kernel_size**2
        kernel *= alpha
        kernel = kernel.repeat(channels, 1, 1).unsqueeze(1)

        tensor_image = image.permute(0, 3, 1, 2) # Torch wants (B, C, H, W) we use (B, H, W, C)
        sharpened = F.conv2d(tensor_image, kernel, padding=center, groups=channels)
        sharpened = sharpened.permute(0, 2, 3, 1)

        result = torch.clamp(sharpened, 0, 1)

        return (result,)

class Solarize:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "solarize_image"

    CATEGORY = "postprocessing"

    def solarize_image(self, image: torch.Tensor, threshold: float):
        solarized_image = torch.where(image > threshold, 1 - image, image)
        solarized_image = torch.clamp(solarized_image, 0, 1)
        return (solarized_image,)

def gaussian_kernel(kernel_size: int, sigma: float):
    x, y = torch.meshgrid(torch.linspace(-1, 1, kernel_size), torch.linspace(-1, 1, kernel_size), indexing="ij")
    d = torch.sqrt(x * x + y * y)
    g = torch.exp(-(d * d) / (2.0 * sigma * sigma))
    return g / g.sum()

def sort_span(span, sort_by, reverse_sorting):
    if sort_by == 'H':
        key = lambda x: x[1][0]
    elif sort_by == 'S':
        key = lambda x: x[1][1]
    else:
        key = lambda x: x[1][2]

    span = sorted(span, key=key, reverse=reverse_sorting)
    return [x[0] for x in span]


def find_spans(mask, span_limit=None):
    spans = []
    start = None
    for i, value in enumerate(mask):
        if value == 0 and start is None:
            start = i
        if value == 1 and start is not None:
            span_length = i - start
            if span_limit is None or span_length <= span_limit:
                spans.append((start, i))
            start = None
    if start is not None:
        span_length = len(mask) - start
        if span_limit is None or span_length <= span_limit:
            spans.append((start, len(mask)))

    return spans


def pixel_sort(img, mask, horizontal_sort=False, span_limit=None, sort_by='H', reverse_sorting=False):
    height, width, _ = img.shape
    hsv_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv_image[..., 0] /= 2.0  # Scale H channel to [0, 1] range

    mask = np.where(mask > 0, 1, 0).astype(np.uint8)

    # loop over the rows and replace contiguous bands of 1s
    for i in range(height if horizontal_sort else width):
        in_band = False
        start = None
        end = None
        for j in range(width if horizontal_sort else height):
            if (mask[i, j] if horizontal_sort else mask[j, i]) == 1:
                if not in_band:
                    in_band = True
                    start = j
                end = j
            else:
                if in_band:
                    for k in range(start+1, end):
                        if horizontal_sort:
                            mask[i, k] = 0
                        else:
                            mask[k, i] = 0
                    in_band = False

        if in_band:
            for k in range(start+1, end):
                if horizontal_sort:
                    mask[i, k] = 0
                else:
                    mask[k, i] = 0

    sorted_image = np.zeros_like(img)
    if horizontal_sort:
        for y in range(height):
            row_mask = mask[y]
            spans = find_spans(row_mask, span_limit)
            sorted_row = np.copy(img[y])
            for start, end in spans:
                span = [(img[y, x], hsv_image[y, x]) for x in range(start, end)]
                sorted_span = sort_span(span, sort_by, reverse_sorting)
                for i, pixel in enumerate(sorted_span):
                    sorted_row[start + i] = pixel
            sorted_image[y] = sorted_row
    else:
        for x in range(width):
            column_mask = mask[:, x]
            spans = find_spans(column_mask, span_limit)
            sorted_column = np.copy(img[:, x])
            for start, end in spans:
                span = [(img[y, x], hsv_image[y, x]) for y in range(start, end)]
                sorted_span = sort_span(span, sort_by, reverse_sorting)
                for i, pixel in enumerate(sorted_span):
                    sorted_column[start + i] = pixel
            sorted_image[:, x] = sorted_column

    return sorted_image

NODE_CLASS_MAPPINGS = {
    "ArithmeticBlend": ArithmeticBlend,
    "Blend": Blend,
    "Blur": Blur,
    "CannyEdgeDetection": CannyEdgeDetection,
    "ColorCorrect": ColorCorrect,
    "Dissolve": Dissolve,
    "DodgeAndBurn": DodgeAndBurn,
    "FilmGrain": FilmGrain,
    "Glow": Glow,
    "PencilSketch": PencilSketch,
    "PixelSort": PixelSort,
    "Pixelize": Pixelize,
    "Quantize": Quantize,
    "Sharpen": Sharpen,
    "Solarize": Solarize,
}
