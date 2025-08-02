from PIL import Image
import numpy as np
import math, random

def preprocess_image(image_path):
    try:
        image = Image.open(image_path)
        image_array = np.array(image)
        
        # Check if image is already grayscale (2D) or color (3D)
        if len(image_array.shape) == 2:
            return image_array
        elif len(image_array.shape) == 3:
            # Convert RGB to Grayscale using the formula: Y = 0.299R + 0.587G + 0.114B
            height, width = image_array.shape[:2]
            image_gray_array = np.zeros((height, width), dtype=np.uint8)
            for i in range(height):
                for j in range(width):
                    r = image_array[i, j, 0]
                    g = image_array[i, j, 1]
                    b = image_array[i, j, 2]
                    image_gray_array[i, j] = int(0.299 * r + 0.587 * g + 0.114 * b)
            return image_gray_array
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def apply_contrast_stretching(image):
    if len(image.shape) != 2:
        raise ValueError("Image must be grayscale (2D)")
    
    image_float = image.astype(np.float32)
    
    i_min = np.min(image_float)
    i_max = np.max(image_float)
    
    print(f"Original image's pixel range: [{i_min}, {i_max}]")
    # edge case: when min == max, return original image
    if i_min == i_max:
        return image
    
    enhanced_image = ((image_float - i_min) / (i_max - i_min)) * 255.0
    
    enhanced_image = np.clip(enhanced_image, 0, 255).astype(np.uint8)
    
    return enhanced_image

image_path = '20701014_Q14/20701014_Q14_input.jpg'
preprocessed_image = preprocess_image(image_path)
if preprocessed_image is not None:
    enhanced_image = apply_contrast_stretching(preprocessed_image)
    enhanced_image_pil = Image.fromarray(enhanced_image)
    enhanced_image_pil.save('20701014_Q14/20701014_Q14_output.jpg')