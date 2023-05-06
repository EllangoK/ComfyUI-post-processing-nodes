import cv2
import torch
import numpy as np

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

    CATEGORY = "postprocessing/Effects"

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

NODE_CLASS_MAPPINGS = {
    "PixelSort": PixelSort,
}

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
