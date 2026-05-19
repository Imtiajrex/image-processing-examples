# Median Filter

## Problem

**Salt-and-pepper noise** corrupts individual pixels by setting them randomly to either 0 (black) or 255 (white), regardless of their true value. The averaging filter struggles here because a single extreme outlier pixel (e.g. 255) pulls the mean far from the true neighbourhood value. The **median filter** is robust to these outliers.

## Concepts

For each pixel, the median filter:
1. Collects all pixel values in a *k × k* neighbourhood.
2. Sorts them.
3. Replaces the centre pixel with the **median** (middle value).

Because the median is insensitive to extreme values, a single salt or pepper pixel in the neighbourhood does not affect the output. Edge sharpness is largely preserved compared to the averaging filter.

**Border handling:** the image is zero-padded by `k // 2` pixels so that border pixels have a full neighbourhood.

## Solution

```
1. Load and convert to grayscale.
2. Inject salt-and-pepper noise (5% of pixels randomly set to 0 or 255).
3. Zero-pad the noisy image by k//2 = 1 pixel.
4. Slide a 3×3 window over every pixel and write the median of the 9 values.
5. Save both the noisy image and the filtered result.
```

```python
neighbors = padded_image[i:i + kernel_size, j:j + kernel_size]
filtered_image[i, j] = np.median(neighbors)
```

### Why median beats mean for salt-and-pepper noise

In a 3×3 window of mostly mid-grey pixels (≈ 128), a single white outlier (255) raises the **mean** noticeably but is simply the 9th value when **sorted** — the median remains the 5th value, unaffected.

## Output

`Median_Filter_noisy.jpg` — the input image with 5% salt-and-pepper noise added.

`Median_Filter_output.jpg` — side-by-side collage:

```
[ Noisy image ] [ Median-filtered image ]
```
