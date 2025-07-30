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

def compute_fft_spectrum(image):
    if len(image.shape) != 2:
        raise ValueError("Image must be grayscale (2D)")
    
    image_float = image.astype(np.float32)
    fft_result = np.fft.fft2(image_float)
    
    # shift the spectrum to center low frequencies
    fft_shifted = np.fft.fftshift(fft_result)
    
    magnitude_spectrum = np.abs(fft_shifted)
    magnitude_spectrum = np.log1p(magnitude_spectrum)  # log(1 + x) to compress dynamic range
    
    # Normalize to [0, 255] for visualization
    spectrum_image = (magnitude_spectrum / np.max(magnitude_spectrum) * 255).astype(np.uint8)
    
    return spectrum_image

image_path = '20701014_Q7/20701014_Q7_input.jpg'
preprocessed_image = preprocess_image(image_path)
if preprocessed_image is not None:
    spectrum_image = compute_fft_spectrum(preprocessed_image)
    spectrum_image_pil = Image.fromarray(spectrum_image)
    spectrum_image_pil.save('20701014_Q7/20701014_Q7_output.jpg')