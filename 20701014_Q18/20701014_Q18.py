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

def binarize_and_close(image):
    if len(image.shape) != 2:
        raise ValueError("Image must be grayscale (2D)")
    
    height, width = image.shape
    
    # binarize
    binary_image = np.zeros((height, width), dtype=np.uint8)
    threshold = 128
    for i in range(height):
        for j in range(width):
            binary_image[i, j] = 255 if image[i, j] >= threshold else 0
    
    struct_elem = np.ones((3, 3), dtype=np.uint8)
    
    # dilation
    dilated_image = np.zeros((height, width), dtype=np.uint8)
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            dilate = False
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    if struct_elem[di + 1, dj + 1] == 1:
                        if binary_image[i + di, j + dj] == 255:
                            dilate = True
                            break
                if dilate:
                    break
            dilated_image[i, j] = 255 if dilate else 0
    
    dilated_image[0, :], dilated_image[height-1, :], dilated_image[:, 0], dilated_image[:, width-1] = 0, 0, 0, 0
    
    # erosion
    closed_image = np.zeros((height, width), dtype=np.uint8)
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if dilated_image[i, j] == 255:
                erode = True
                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        if struct_elem[di + 1, dj + 1] == 1:
                            if dilated_image[i + di, j + dj] == 0:
                                erode = False
                                break
                    if not erode:
                        break
                closed_image[i, j] = 255 if erode else 0
            else:
                closed_image[i, j] = 0
    
    closed_image[0, :], closed_image[height-1, :], closed_image[:, 0], closed_image[:, width-1] = 0, 0, 0, 0
    
    return closed_image

image_path = '20701014_Q18/20701014_Q18_input.jpg'
preprocessed_image = preprocess_image(image_path)
if preprocessed_image is not None:
    closed_image = binarize_and_close(preprocessed_image)
    closed_image_pil = Image.fromarray(closed_image)
    closed_image_pil.save('20701014_Q18/20701014_Q18_output.jpg')