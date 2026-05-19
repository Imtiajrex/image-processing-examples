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

def add_constant(image, value):
    if len(image.shape) != 2:
        raise ValueError("Image must be grayscale (2D)")
    height, width = image.shape
    result = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            result[i, j] = min(255, max(0, int(image[i, j]) + value))
    return result

def subtract_constant(image, value):
    if len(image.shape) != 2:
        raise ValueError("Image must be grayscale (2D)")
    height, width = image.shape
    result = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            result[i, j] = min(255, max(0, int(image[i, j]) - value))
    return result

def multiply_constant(image, scalar):
    if len(image.shape) != 2:
        raise ValueError("Image must be grayscale (2D)")
    height, width = image.shape
    result = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            result[i, j] = min(255, max(0, int(image[i, j] * scalar)))
    return result

image_path = 'Arithmetic_Operation/Arithmetic_Operation_input.jpg'
preprocessed_image = preprocess_image(image_path)
if preprocessed_image is not None:
    added = add_constant(preprocessed_image, 50)         # Addition: increases brightness
    subtracted = subtract_constant(preprocessed_image, 50)  # Subtraction: decreases brightness
    multiplied = multiply_constant(preprocessed_image, 1.5)  # Multiplication: scales intensity

    # Create collage: original | added | subtracted | multiplied
    height, width = preprocessed_image.shape
    collage = np.zeros((height, width * 4), dtype=np.uint8)
    collage[:, :width] = preprocessed_image
    collage[:, width:width * 2] = added
    collage[:, width * 2:width * 3] = subtracted
    collage[:, width * 3:] = multiplied

    collage_pil = Image.fromarray(collage)
    collage_pil.save('Arithmetic_Operation/Arithmetic_Operation_output.jpg')
