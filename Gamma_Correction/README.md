# Gamma Correction

## Problem

Human vision perceives brightness non-linearly — we are more sensitive to changes in dark tones than bright tones. Display devices also have a non-linear response to input signals. **Gamma correction** applies a power-law transformation to compensate for these effects or to intentionally brighten/darken an image.

## Concepts

The transformation is applied per pixel after normalising the intensity to [0, 1]:

```
output = (input / 255)^γ × 255
```

The behaviour depends on the value of γ:

| γ | Effect |
|---|---|
| γ < 1 | Brightens the image (lifts shadows) |
| γ = 1 | No change |
| γ > 1 | Darkens the image (crushes highlights) |

This is a **point operation** — each pixel is transformed independently with no spatial context.

## Solution

```
1. Load and convert to grayscale.
2. For each pixel, normalise to [0.0, 1.0].
3. Raise to the power γ.
4. Scale back to [0, 255] and cast to uint8.
```

```python
corrected[i, j] = int((image[i, j] / 255.0) ** gamma * 255)
```

Two variants are demonstrated:

- `gamma = 0.5` → brightened image (square-root curve)
- `gamma = 2.0` → darkened image (square curve)

## Output

`Gamma_Correction_output.jpg` — a three-panel collage:

```
[ Original ] [ γ = 0.5 (brighter) ] [ γ = 2.0 (darker) ]
```
