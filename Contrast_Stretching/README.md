# Contrast Stretching

## Problem

Images from poorly lit environments or low-quality sensors often have a narrow pixel intensity range — for example, all pixels sitting between 80 and 160 rather than spanning the full 0–255 range. This makes the image appear washed out or low-contrast. Contrast stretching remaps the existing range to the full 8-bit range to reveal hidden detail.

## Concepts

**Contrast stretching** (also called **min-max normalisation**) is a linear point operation. Given the minimum intensity `i_min` and maximum intensity `i_max` of the image, every pixel value `p` is remapped as:

```
output = (p − i_min) / (i_max − i_min) × 255
```

- A pixel at `i_min` maps to 0 (pure black).
- A pixel at `i_max` maps to 255 (pure white).
- All other values are linearly distributed in between.

This is different from **histogram equalisation**, which applies a non-linear mapping to achieve a uniform distribution.

## Solution

```
1. Load and convert to grayscale.
2. Find i_min and i_max with np.min / np.max.
3. Guard against the degenerate case where i_min == i_max (flat image).
4. Apply the normalisation formula.
5. Clip to [0, 255] and cast to uint8.
```

```python
enhanced = ((image_float - i_min) / (i_max - i_min)) * 255.0
enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
```

The cast to `float32` before arithmetic prevents integer truncation errors during division.

### Edge case

If `i_min == i_max` (all pixels are identical), the formula would divide by zero. The implementation returns the original image unchanged in that case.

## Output

`Contrast_Stretching_output.jpg` — the same scene with a full 0–255 tonal range, showing improved local detail and higher perceived contrast.
