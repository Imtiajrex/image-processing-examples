from PIL import Image
import numpy as np

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

def apply_sharpening_filter(image, kernel_size = 3):
    if len(image.shape) != 2:
        raise ValueError("Image must be grayscale (2D)")

    height, width = image.shape
    # creating the sharpening filter kernel
    kernel = np.array([[-1, -1, -1],
                       [-1, 8, -1],
                       [-1, -1, -1]], dtype=np.float32)

    # padding the image to handle borders
    pad_amount = kernel_size // 2
    padded_image = np.pad(image, pad_amount, mode='constant', constant_values=0)

    # applying the averaging filter
    filtered_image = np.zeros_like(image, dtype=np.float32)
    for i in range(height):
        for j in range(width):
            neighbors = padded_image[i:i + kernel_size, j:j + kernel_size]
            filtered_image[i, j] = np.sum(neighbors * kernel)
            filtered_image[i, j] = np.sum(neighbors * kernel) 
    output = np.clip(filtered_image, 0, 255).astype(np.uint8)
    return output

preprocessed_image = preprocess_image('20701014_Q3/20701014_Q3_input.jpg')
filtered_image = apply_sharpening_filter(preprocessed_image)

filtered_image_pil = Image.fromarray(np.uint8(filtered_image))
filtered_image_pil.save('20701014_Q3/20701014_Q3_output.jpg')