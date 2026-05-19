# Histogram Equalization

## Problem

An image with poor global contrast has pixel intensities clustered in a small region of the 0–255 range. **Histogram equalisation** redistributes intensities so they are spread more uniformly across the full range, enhancing contrast without requiring any manual parameter tuning.

## Concepts

The algorithm works through three steps:

1. **Histogram** — count how many pixels have each intensity value (0–255).
2. **Cumulative Distribution Function (CDF)** — accumulate the histogram so that `CDF[k]` is the number of pixels with intensity ≤ k.
3. **Mapping** — normalise the CDF to [0, 255] and use it as a lookup table:

```
output[p] = round( (CDF[p] − CDF_min) / (total_pixels − CDF_min) × 255 )
```

Pixels in dense regions of the histogram get spread out; sparse regions contract, producing a flatter output histogram and more visually distinct tones.

## Solution

This implementation operates on an **RGB image** (not grayscale):

```
1. Convert RGB to grayscale manually to compute the intensity histogram.
2. Build the histogram by counting pixel occurrences.
3. Compute the CDF by cumulative summation of the histogram.
4. Normalise the CDF to [0, 255].
5. For each pixel, compute the ratio between the equalized gray value
   and the original gray value, then scale all three RGB channels by
   that factor to preserve colour balance.
6. Clip all channel values to [0, 255].
```

### Colour preservation strategy

Rather than equalising each channel independently (which distorts hues), the intensity ratio is applied uniformly:

```python
factor = equalized_gray[i, j] / original_gray[i, j]
R_out = R_in * factor
G_out = G_in * factor
B_out = B_in * factor
```

This keeps relative colour relationships intact while adjusting overall brightness.

## Output

`Histogram_Equalization_output.jpg` — the colour image with improved global contrast and a more uniform intensity distribution.
