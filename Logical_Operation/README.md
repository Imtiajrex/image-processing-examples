# Logical Operation

## Problem

Bitwise operations on pixel values are fundamental building blocks for image masking, region extraction, and combining binary or grayscale images. Understanding how AND, OR, NOT, and XOR behave on pixel data is essential for tasks like background removal and feature highlighting.

## Concepts

Each pixel is an 8-bit integer (0–255). Bitwise operations act on every bit of that value independently:

| Operation | Pixel formula | Effect |
|---|---|---|
| AND (`&`) | `A & B` | Keeps only bits set in **both** images — used for masking |
| OR (`\|`) | `A \| B` | Sets bits present in **either** image — used for merging |
| NOT (`~`) | `255 − A` | Inverts every bit — produces the negative of the image |
| XOR (`^`) | `A ^ B` | Sets bits present in **one but not both** — highlights differences |

Note: NOT is implemented as `255 − pixel` rather than Python's `~` operator, which would produce negative values for uint8.

## Solution

A synthetic **binary mask** is created to make the operations visually clear:
- Left half: 255 (white)
- Right half: 0 (black)

The four operations are then applied between the grayscale image and this mask:

```python
mask[:, :width // 2] = 255   # left half white, right half black

and_result = image & mask    # left half of image preserved, right half blacked out
or_result  = image | mask    # left half all white, right half unchanged
not_result = 255 - image     # full inverted image (ignores mask)
xor_result = image ^ mask    # left half inverted, right half unchanged
```

## Output

`Logical_Operation_output.jpg` — a four-panel collage:

```
[ AND ] [ OR ] [ NOT ] [ XOR ]
```
