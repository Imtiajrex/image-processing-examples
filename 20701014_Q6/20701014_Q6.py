from PIL import Image
import numpy as np

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

def apply_sharpening_filter(image, kernel_size=3):
    if len(image.shape) != 2:
        raise ValueError("Image must be grayscale (2D)")
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd")
    
    height, width = image.shape
    
    # create custom sharpening kernel
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = kernel_size // 2

    for i in range(kernel_size//2 + 1):
        kernel[i, center - i: center + i + 1] = -1
    for i in range(kernel_size//2 + 1, kernel_size):
        kernel[i, :] = kernel[kernel_size - i - 1, :]
    kernel[center, center] = 0

    # set center value to make kernel sum to 1
    kernel[center, center] = abs(np.sum(kernel))
    print(kernel)
    
    # pad the image with zeros
    pad_amount = kernel_size // 2
    padded_image = np.pad(image, pad_amount, mode='constant', constant_values=0)
    
    sharpened_image = np.zeros_like(image, dtype=np.float32)
    for i in range(height):
        for j in range(width):
            region = padded_image[i:i+kernel_size, j:j+kernel_size]
            sharpened_image[i, j] = np.sum(region * kernel)
    
    sharpened_image = np.clip(sharpened_image, 0, 255).astype(np.uint8)
    
    return sharpened_image

image_path = '20701014_Q6/20701014_Q6_input.jpg'
preprocessed_image = preprocess_image(image_path)
if preprocessed_image is not None:
    sharpened_image = apply_sharpening_filter(preprocessed_image, kernel_size=5)
    sharpened_image_pil = Image.fromarray(sharpened_image)
    sharpened_image_pil.save('20701014_Q6/20701014_Q6_output.jpg')