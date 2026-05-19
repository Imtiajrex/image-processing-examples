# Bilinear Interpolation

## Problem

Resizing an image requires computing pixel values at non-integer coordinates in the source image. The simplest approach — **nearest-neighbour** — copies the closest pixel, producing a blocky result. **Bilinear interpolation** produces a smoother result by blending the four surrounding source pixels.

## Concepts

When scaling by factors `scale_x` and `scale_y`, each destination pixel `(i, j)` maps back to a floating-point coordinate in the source:

```
src_x = j / scale_x
src_y = i / scale_y
```

The four nearest source pixels form a 2 × 2 neighbourhood:

```
(x0, y0)  (x1, y0)
(x0, y1)  (x1, y1)
```

where `x0 = floor(src_x)`, `x1 = x0 + 1`, and similarly for y.

The fractional parts `a = src_x − x0` and `b = src_y − y0` act as weights:

```
output = (1−a)(1−b)·P(y0,x0)
       +    a (1−b)·P(y0,x1)
       + (1−a)  b  ·P(y1,x0)
       +    a   b  ·P(y1,x1)
```

This is a weighted average: pixels closer to the target point contribute more.

## Solution

```
1. Load and convert to grayscale.
2. Compute destination dimensions from scale factors.
3. For each destination pixel, reverse-map to source coordinates.
4. Clamp neighbour indices to image bounds (handles right/bottom edges).
5. Apply the bilinear formula and write the integer result.
```

Scale factors used: `scale_x = 1.5`, `scale_y = 1.5` (50% enlargement in both axes).

### Edge clamping

```python
x1 = min(x0 + 1, src_width - 1)
y1 = min(y0 + 1, src_height - 1)
```

This prevents out-of-bounds access when a source coordinate lands exactly on the last row or column.

## Output

`Bilinear_Interpolation_output.jpg` — the input image scaled to 1.5× its original size with smooth, artifact-free edges.
