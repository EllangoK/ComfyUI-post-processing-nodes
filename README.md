#  ComfyUI-post-processing-nodes

A collection of post processing nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI), simply download this repo and drag `post_processing_nodes.py` into your `custom_nodes/` folder

## Node List

 - Blend: Blends two images together with a variety of different modes
 - CannyEdgeDetection: Applies Canny edge detection to the input image
 - ColorCorrect: Adjusts the color balance, temperature, hue, brightness, contrast, saturation, and gamma of an image
 - Dither: Reduces the color information in an image by dithering, resulting in a patterned, pixelated appearance
 - FilmGrain: Adds a film grain effect to the image, along with options to control the temperature, and vignetting
 - Glow: Applies a blur with a specified radius and then blends it with the original image. Creates a nice glowing effect.
 - GaussianBlur: Applies a Gaussian blur to the input image, softening the details
 - KMeansQuantize: Reduce the amount of colors in an image from 0-256
 - PixelSort: Rearranges the pixels in the input image based on their values, and input mask. Creates a cool glitch like effect.
 - Sharpen: Enhances the details in an image by applying a sharpening filter

## Example workflow

![__image__](images/example-workflow.png)

## Combine Nodes

By default `post_processing_nodes.py` should have all of the combined nodes. If you want a subset of nodes, you can run

    python combine_files.py [--files FILES [FILES ...]] [--output OUTPUT]

or just run

    python combine_files.py -h for more help