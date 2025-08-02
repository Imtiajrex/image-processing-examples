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

def apply_butterworth_lowpass_filter(image, cutoff_radius=None, order=2):
    if len(image.shape) != 2:
        raise ValueError("Image must be grayscale (2D)")
    
    height, width = image.shape
    
    # Set default cutoff radius if not provided
    if cutoff_radius is None:
        cutoff_radius = min(height, width) / 4
    
    image_float = image.astype(np.float32)
    
    fft_result = np.fft.fft2(image_float)
    fft_shifted = np.fft.fftshift(fft_result)
    
    mask = np.zeros((height, width), dtype=np.float32)
    center_y, center_x = height // 2, width // 2
    
    for i in range(height):
        for j in range(width):
            distance = math.sqrt((i - center_y)**2 + (j - center_x)**2)
            # butterworth filter response
            mask[i, j] = 1.0 / (1.0 + (distance / cutoff_radius)**(2 * order))
    
    fft_filtered = fft_shifted * mask
    
    fft_ishifted = np.fft.ifftshift(fft_filtered)
    filtered_image = np.fft.ifft2(fft_ishifted)
    
    filtered_image = np.real(filtered_image)
    filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)
    
    return filtered_image

image_path = '20701014_Q9/20701014_Q9_input.jpg'
preprocessed_image = preprocess_image(image_path)
if preprocessed_image is not None:
    filtered_image = apply_butterworth_lowpass_filter(preprocessed_image, cutoff_radius=None, order=2)
    filtered_image_pil = Image.fromarray(filtered_image)
    filtered_image_pil.save('20701014_Q9/20701014_Q9_output.jpg')