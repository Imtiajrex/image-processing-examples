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

def binarize_and_remove_small_objects(image, min_size=50):
    if len(image.shape) != 2:
        raise ValueError("Image must be grayscale (2D)")
    
    height, width = image.shape
    
    # binarize
    binary_image = np.zeros((height, width), dtype=np.uint8)
    threshold = 128
    for i in range(height):
        for j in range(width):
            binary_image[i, j] = 255 if image[i, j] >= threshold else 0
    
    # connected component labeling
    label = 0
    labels = np.zeros((height, width), dtype=np.int32)
    object_sizes = {}
    
    for i in range(height):
        for j in range(width):
            if binary_image[i, j] == 255 and labels[i, j] == 0:
                label += 1
                size = 0
                stack = [(i, j)]
                while stack:
                    x, y = stack.pop()
                    if 0 <= x < height and 0 <= y < width and binary_image[x, y] == 255 and labels[x, y] == 0:
                        labels[x, y] = label
                        size += 1
                        # Check 4-connectivity (up, down, left, right)
                        stack.append((x-1, y))
                        stack.append((x+1, y))
                        stack.append((x, y-1))
                        stack.append((x, y+1))
                object_sizes[label] = size
    
    cleaned_image = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            if labels[i, j] > 0 and object_sizes.get(labels[i, j], 0) >= min_size:
                cleaned_image[i, j] = 255
    
    return cleaned_image

image_path = '20701014_Q19/20701014_Q19_input.jpg'
preprocessed_image = preprocess_image(image_path)
if preprocessed_image is not None:
    cleaned_image = binarize_and_remove_small_objects(preprocessed_image, min_size=100)
    cleaned_image_pil = Image.fromarray(cleaned_image)
    cleaned_image_pil.save('20701014_Q19/20701014_Q19_output.jpg')