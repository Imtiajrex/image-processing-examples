# Averaging Filter

## Problem

Images captured in low-light conditions or with sensor noise contain random pixel variations that obscure the true content. A simple way to reduce this high-frequency noise is spatial smoothing — replacing each pixel with an average of its local neighbourhood.

## Concepts

The **averaging filter** (also called a box filter) is a linear spatial filter. For a kernel of size *k × k*, each output pixel is the mean of the *k²* pixels centred on that location in the input:

```
output[i, j] = (1 / k²) · Σ input[i+p, j+q]   for p, q in [−k/2, k/2]
```

The kernel is uniform — every neighbour contributes equally — which blurs the image and attenuates noise at the cost of sharpness.

**Border handling:** the image is zero-padded by `k // 2` pixels on every side so that the filter can be applied to edge pixels without going out of bounds.

## Solution

```
1. Load and convert the image to grayscale.
2. Construct a k×k kernel filled with 1/(k²).
3. Zero-pad the image by k//2 pixels.
4. Slide the kernel over every pixel and compute the weighted sum.
5. Save the result.
```

The kernel used here is 3 × 3:

```python
kernel = np.ones((3, 3), dtype=np.float32) / 9
```

### Gaussian noise injection (optional)

The script includes an `add_gaussian_noise` helper (commented out) that adds normally-distributed noise (mean=0, std=30) to simulate a degraded input before filtering.

## Output

`Averaging_Filter_output.jpg` — the smoothed image with reduced noise and slightly softened edges compared to the input.
