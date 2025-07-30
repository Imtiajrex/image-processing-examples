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

def apply_ideal_lowpass_filter(image, cutoff_radius=None):
    if len(image.shape) != 2:
        raise ValueError("Image must be grayscale (2D)")
    
    height, width = image.shape
    
    if cutoff_radius is None:
        cutoff_radius = min(height, width) / 6
    
    # Convert to float32 for FFT
    image_float = image.astype(np.float32)
    
    # Compute 2D FFT and shift to center low frequencies
    fft_result = np.fft.fft2(image_float)
    fft_shifted = np.fft.fftshift(fft_result)
    
    # Create ideal low-pass filter mask
    mask = np.zeros((height, width), dtype=np.float32)
    center_y, center_x = height // 2, width // 2
    
    for i in range(height):
        for j in range(width):
            # Compute distance from center
            distance = math.sqrt((i - center_y)**2 + (j - center_x)**2)
            if distance <= cutoff_radius:
                mask[i, j] = 1.0
    
    # Apply the filter
    fft_filtered = fft_shifted * mask
    
    # Inverse FFT and shift back
    fft_ishifted = np.fft.ifftshift(fft_filtered)
    filtered_image = np.fft.ifft2(fft_ishifted)
    
    # Take the real part and normalize to [0, 255]
    filtered_image = np.real(filtered_image)
    filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)
    
    return filtered_image

image_path = '20701014_Q8/20701014_Q8_input.jpg'
preprocessed_image = preprocess_image(image_path)
if preprocessed_image is not None:
    filtered_image = apply_ideal_lowpass_filter(preprocessed_image, cutoff_radius=None)
    filtered_image_pil = Image.fromarray(filtered_image)
    filtered_image_pil.save('20701014_Q8/20701014_Q8_lowpass_filtered.jpg')