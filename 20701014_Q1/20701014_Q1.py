from PIL import Image
import numpy as np

def add_gaussian_noise(image, mean=0, std=30):
    gaussian_noise = np.random.normal(mean, std, image.shape)
    noisy_image = image.astype(np.float32) + gaussian_noise
    # Clip values to valid range [0, 255] and convert back to uint8
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
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
            # Color image, convert to grayscale
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

def apply_averaging_filter(image, kernel_size):
    if len(image.shape) != 2:
        raise ValueError("Image must be grayscale (2D)")
    height, width = image.shape

    if(kernel_size % 2 == 0):
        raise ValueError("Kernel size must be an odd number.")
    # Creating the averaging filter kernel
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)

    # Padding the image to handle borders
    pad_amount = kernel_size // 2
    padded_image = np.pad(image, pad_amount, mode='constant', constant_values=0)

    # Applying the averaging filter
    filtered_image = np.zeros_like(image, dtype=np.float32)
    for i in range(height):
        for j in range(width):
            neighbors = padded_image[i:i + kernel_size, j:j + kernel_size]
            filtered_image[i, j] = np.sum(neighbors * kernel)
    return filtered_image

preprocessed_image = preprocess_image(image_path='20701014_Q1/20701014_Q1_input.jpg')
# noisy_image = add_gaussian_noise(preprocessed_image)
# noisy_image_pil = Image.fromarray(np.uint8(noisy_image))
# noisy_image_pil.save('20701014_Q1/20701014_Q1_input.jpg')
filtered_image = apply_averaging_filter(preprocessed_image, kernel_size=3)

filtered_image_pil = Image.fromarray(np.uint8(filtered_image))
filtered_image_pil.save('20701014_Q1/20701014_Q1_output.jpg')