# Arithmetic Operation

## Problem

Images often need their overall brightness adjusted — making them lighter, darker, or changing the contrast by scaling pixel intensities. Raw pixel operations must stay within the valid 8-bit range [0, 255], so naive addition or multiplication can overflow or underflow.

## Concepts

Arithmetic operations treat each pixel as a number and apply a mathematical transformation:

- **Addition** — shifts all intensities up, brightening the image.
- **Subtraction** — shifts all intensities down, darkening the image.
- **Multiplication** — scales intensities, expanding (>1) or compressing (<1) the dynamic range.

Every result is clamped with `min(255, max(0, value))` to prevent wrap-around artifacts.

## Solution

The script converts the input image to grayscale using the luminance formula:

```
Y = 0.299·R + 0.587·G + 0.114·B
```

Then it applies three operations to the grayscale image:

| Operation | Parameter | Effect |
|---|---|---|
| `add_constant` | +50 | Increases brightness |
| `subtract_constant` | −50 | Decreases brightness |
| `multiply_constant` | ×1.5 | Scales intensity (higher contrast) |

### Key implementation detail

Each pixel is cast to `int` before arithmetic to avoid NumPy uint8 overflow before the clamp is applied:

```python
result[i, j] = min(255, max(0, int(image[i, j]) + value))
```

## Output

The four images (original, added, subtracted, multiplied) are stitched side-by-side into a single collage saved as `Arithmetic_Operation_output.jpg`.

```
[ Original ] [ +50 ] [ −50 ] [ ×1.5 ]
```
