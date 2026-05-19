from PIL import Image
import numpy as np
import math, random

def add_salt_pepper_noise(image, noise_prob=0.05):
    height, width = image.shape[:2]
    channels = 1 if len(image.shape) == 2 else image.shape[2]
    
    noisy_image = image.copy()
    
    total_pixels = height * width
    num_noisy = int(noise_prob * total_pixels)
    
    for _ in range(num_noisy):
        i = random.randint(0, height - 1)
        j = random.randint(0, width - 1)
        
        if random.random() < 0.5:
            value = 255  # salt
        else:
            value = 0    # pepper
        
        # apply to all channels for color images or single value for grayscale
        if channels > 1:
            noisy_image[i, j, :] = value
        else:
            noisy_image[i, j] = value
    
    return noisy_image

def preprocess_image(image_path):
    try:
        image = Image.open(image_path)
        # image_gray = image.convert("L")
        image_array = np.array(image)
        
        # check if image is already grayscale (2D) or color (3D)
        if len(image_array.shape) == 2:
            return image_array
        elif len(image_array.shape) == 3:
            # color image, convert to grayscale
            height, width = image_array.shape[:2]
            image_gray_array = np.zeros((height, width), dtype=np.uint8)
            # Convert RGB to Grayscale using the formula: Y = 0.299R + 0.587G + 0.114B
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

def apply_median_filter(image, kernel_size=3):
    if len(image.shape) != 2:
        raise ValueError("Image must be grayscale (2D)")

    height, width = image.shape
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be an odd number.")

    # padding the image to handle borders
    pad_amount = kernel_size // 2
    padded_image = np.pad(image, pad_amount, mode='constant', constant_values=0)

    # applying the median filter
    filtered_image = np.zeros_like(image, dtype=np.float32)
    for i in range(height):
        for j in range(width):
            neighbors = padded_image[i:i + kernel_size, j:j + kernel_size]
            median_value = np.median(neighbors)
            filtered_image[i, j] = median_value
    return filtered_image

preprocessed_image = preprocess_image(image_path='Median_Filter/Median_Filter_input.jpg')
noisy_image = add_salt_pepper_noise(preprocessed_image)
noisy_image_pil = Image.fromarray(np.uint8(noisy_image))
noisy_image_pil.save('Median_Filter/Median_Filter_noisy.jpg')
filtered_image = apply_median_filter(noisy_image, kernel_size=3)

# collage of noisy and filtered images
collage_width = noisy_image.shape[1] + filtered_image.shape[1]
collage_height = max(noisy_image.shape[0], filtered_image.shape[0])
collage = np.zeros((collage_height, collage_width), dtype=np.uint8)
collage[:noisy_image.shape[0], :noisy_image.shape[1]] = noisy_image
collage[:filtered_image.shape[0], noisy_image.shape[1]:] = filtered_image
collage_pil = Image.fromarray(np.uint8(collage))
collage_pil.save('Median_Filter/Median_Filter_output.jpg')