from PIL import Image
import numpy as np
import math, random

def preprocess_image(image_path):
    try:
        image = Image.open(image_path)
        image_array = np.array(image)
        return image_array 
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def apply_histogram_equalization_rgb_scratch(image):
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Image must be RGB (3 channels)")
    
    height, width = image.shape[:2]
    
    # Convert to grayscale manually
    gray_image = np.zeros((height, width), dtype=np.float32)
    for i in range(height):
        for j in range(width):
            r, g, b = image[i, j, 0], image[i, j, 1], image[i, j, 2]
            gray_image[i, j] = 0.299 * r + 0.587 * g + 0.114 * b
    
    # Compute histogram manually
    hist = np.zeros(256, dtype=np.int32)
    for i in range(height):
        for j in range(width):
            intensity = int(gray_image[i, j])
            hist[intensity] += 1
    
    # Compute CDF manually
    cdf = np.zeros(256, dtype=np.float32)
    cdf[0] = hist[0]
    for i in range(1, 256):
        cdf[i] = cdf[i-1] + hist[i]
    
    # Normalize CDF to [0, 255]
    cdf_min = cdf[np.nonzero(cdf)[0][0]] if np.any(cdf) else 0
    cdf_max = cdf[-1]
    if cdf_max > cdf_min:
        for i in range(256):
            if cdf[i] != 0:
                cdf[i] = ((cdf[i] - cdf_min) * 255) / (cdf_max - cdf_min)
    
    # Apply histogram equalization to grayscale
    equalized_gray = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            original_intensity = int(gray_image[i, j])
            equalized_gray[i, j] = int(cdf[original_intensity] + 0.5)  # Round to nearest integer
    
    # Map equalized grayscale to RGB channels proportionally
    enhanced_image = image.copy().astype(np.float32)
    for i in range(height):
        for j in range(width):
            original_gray = gray_image[i, j]
            if original_gray != 0: 
                factor = equalized_gray[i, j] / original_gray
                enhanced_image[i, j, 0] = min(255, max(0, image[i, j, 0] * factor))  # Red
                enhanced_image[i, j, 1] = min(255, max(0, image[i, j, 1] * factor))  # Green
                enhanced_image[i, j, 2] = min(255, max(0, image[i, j, 2] * factor))  # Blue
    
    return enhanced_image.astype(np.uint8)

image_path = '20701014_Q15/20701014_Q15_input.jpg'
preprocessed_image = preprocess_image(image_path)
if preprocessed_image is not None:
    enhanced_image = apply_histogram_equalization_rgb_scratch(preprocessed_image)
    enhanced_image_pil = Image.fromarray(enhanced_image)
    enhanced_image_pil.save('20701014_Q15/20701014_Q15_output.jpg')