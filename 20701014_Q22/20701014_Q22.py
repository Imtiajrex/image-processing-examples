from PIL import Image
import numpy as np
import math, random

def rgb2hsv(image):
    height, width, _ = image.shape
    hsv = np.zeros((height, width, 3), dtype=np.float32)
    for i in range(height):
        for j in range(width):
            r, g, b = image[i, j]
            r, g, b = r / 255.0, g / 255.0, b / 255.0
            mx = max(r, g, b)
            mn = min(r, g, b)
            df = mx - mn
            
            # Value
            v = mx
            if mx == 0:
                s = 0
                h = 0
            else:
                # Saturation
                s = df / mx
                # Hue
                if df == 0:
                    h = 0
                elif mx == r:
                    h = (60 * ((g - b) / df) + 360) % 360
                elif mx == g:
                    h = (60 * ((b - r) / df) + 120) % 360
                elif mx == b:
                    h = (60 * ((r - g) / df) + 240) % 360
                h /= 360.0  # Normalize to [0, 1]
            
            hsv[i, j] = [h, s, v]
    return hsv

def rgb2gray(image):
    height, width, _ = image.shape
    gray = np.zeros((height, width), dtype=np.float32)
    for i in range(height):
        for j in range(width):
            r, g, b = image[i, j]
            gray[i, j] = 0.299 * r + 0.587 * g + 0.114 * b
    return gray / 255.0

def sobel(image, axis):
    height, width = image.shape
    sobel_out = np.zeros((height, width), dtype=np.float32)
    if axis == 0:  # Horizontal
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                sobel_out[i, j] = (-image[i-1, j-1] - 2*image[i, j-1] - image[i+1, j-1] +
                                  image[i-1, j+1] + 2*image[i, j+1] + image[i+1, j+1]) / 8.0
    elif axis == 1:  # Vertical
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                sobel_out[i, j] = (-image[i-1, j-1] - 2*image[i-1, j] - image[i-1, j+1] +
                                  image[i+1, j-1] + 2*image[i+1, j] + image[i+1, j+1]) / 8.0
    return sobel_out

def local_binary_pattern(image, P=8, R=1, method="uniform"):
    height, width = image.shape
    lbp = np.zeros((height, width), dtype=np.uint32)
    for i in range(R, height - R):
        for j in range(R, width - R):
            center = image[i, j]
            code = 0
            for p in range(P):
                angle = 2 * math.pi * p / P
                ni = i + int(R * math.sin(angle) + 0.5)
                nj = j + int(R * math.cos(angle) + 0.5)
                code |= (1 << p) if image[ni, nj] > center else 0
            lbp[i, j] = code
    return lbp

def generate_image_descriptor(image_path):
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image).astype(np.float32) / 255.0
    
    descriptor = {}
    
    # 1. Color Histogram in HSV
    hsv_image = rgb2hsv(image_np)
    h_vals = np.histogram(hsv_image[:, :, 0], bins=16, range=(0, 1))[0]
    s_vals = np.histogram(hsv_image[:, :, 1], bins=16, range=(0, 1))[0]
    v_vals = np.histogram(hsv_image[:, :, 2], bins=16, range=(0, 1))[0]
    color_hist = np.concatenate([h_vals, s_vals, v_vals]).astype(np.float32)
    color_hist /= (np.sum(color_hist) + 1e-8)  # Normalize
    descriptor['color_histogram'] = color_hist
    
    # 2. Edge Density using Sobel filter
    gray_image = rgb2gray(image_np)
    sobel_x = sobel(gray_image, axis=0)
    sobel_y = sobel(gray_image, axis=1)
    edge_magnitude = np.hypot(sobel_x, sobel_y)
    edge_density = np.sum(edge_magnitude > 0.2) / edge_magnitude.size  # Thresholded
    descriptor['edge_density'] = edge_density
    
    # 3. Texture Descriptor using LBP
    lbp = local_binary_pattern(gray_image, P=8, R=1, method="uniform")
    lbp_hist = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))[0].astype(np.float32)
    lbp_hist /= (lbp_hist.sum() + 1e-8)  # Normalize
    descriptor['lbp_texture'] = lbp_hist
    
    return descriptor

image_path = '20701014_Q22/20701014_Q22_input.jpg'
descriptor = generate_image_descriptor(image_path)

# print descriptor
print("Image Descriptor:")
print(f"Color Histogram (H:16, S:16, V:16 bins): {descriptor['color_histogram']}")
print(f"Edge Density: {descriptor['edge_density']}")
print(f"LBP Texture (9 bins): {descriptor['lbp_texture']}")

# save descriptor to a text file
with open('20701014_Q22/20701014_Q22_descriptor.txt', 'w') as f:
    f.write("Image Descriptor:\n")
    f.write(f"Color Histogram (H:16, S:16, V:16 bins): {descriptor['color_histogram'].tolist()}\n")
    f.write(f"Edge Density: {descriptor['edge_density']}\n")
    f.write(f"LBP Texture (9 bins): {descriptor['lbp_texture'].tolist()}\n")