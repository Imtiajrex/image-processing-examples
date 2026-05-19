# Sobel Operation

## Problem

Edge detection identifies boundaries between regions in an image — locations where pixel intensity changes sharply. Detecting these edges is a foundational step in object recognition, segmentation, and feature extraction.

## Concepts

The **Sobel operator** estimates the image gradient using two 3×3 convolution kernels. The gradient measures how quickly intensity changes in a given direction.

**Horizontal gradient kernel** (detects vertical edges — intensity changes left-to-right):

```
Gx = [[-1,  0,  1],
      [-2,  0,  2],
      [-1,  0,  1]]
```

**Vertical gradient kernel** (detects horizontal edges — intensity changes top-to-bottom):

```
Gy = [[ 1,  2,  1],
      [ 0,  0,  0],
      [-1, -2, -1]]
```

Each kernel is convolved with the image to produce a gradient map. The **combined edge magnitude** is:

```
G = sqrt(Gx² + Gy²)
```

This is normalised to [0, 255] for display.

## Solution

```
1. Load and convert to grayscale.
2. Zero-pad the image by 1 pixel (kernel_size // 2).
3. Slide each 3×3 kernel over the image and compute the dot product
   (element-wise multiply then sum) to get Gx and Gy.
4. Clip Gx and Gy to [0, 255] for individual edge images.
5. Compute G = sqrt(Gx² + Gy²) and normalise to [0, 255] for the combined map.
```

```python
# Single direction
filtered_image[i, j] = np.sum(neighbors * kernel)

# Combined magnitude
G_magnitude = np.sqrt(Gx**2 + Gy**2)
G_magnitude = (G_magnitude - min_val) / (max_val - min_val) * 255
```

**Note on kernel naming:** the script's `vertical=True` argument applies the `Gx` kernel, which highlights *vertical* edges (the kernel is sensitive to left-right intensity change). `vertical=False` applies `Gy`, highlighting *horizontal* edges.

## Output

`Sobel_Operation_vertical_output.jpg` — edges along vertical boundaries (Gx kernel).

`Sobel_Operation_horizontal_output.jpg` — edges along horizontal boundaries (Gy kernel).

`Sobel_Operation_output.jpg` — side-by-side collage:

```
[ Vertical edges (Gx) ] [ Horizontal edges (Gy) ]
```
