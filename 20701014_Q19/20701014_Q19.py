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

def apply_logical_and(image1, image2):
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same dimensions")
    height, width = image1.shape
    result = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            result[i, j] = image1[i, j] & image2[i, j]
    return result

def apply_logical_or(image1, image2):
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same dimensions")
    height, width = image1.shape
    result = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            result[i, j] = image1[i, j] | image2[i, j]
    return result

def apply_logical_not(image):
    if len(image.shape) != 2:
        raise ValueError("Image must be grayscale (2D)")
    height, width = image.shape
    result = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            result[i, j] = 255 - image[i, j]
    return result

def apply_logical_xor(image1, image2):
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same dimensions")
    height, width = image1.shape
    result = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            result[i, j] = image1[i, j] ^ image2[i, j]
    return result

image_path = '20701014_Q19/20701014_Q19_input.jpg'
preprocessed_image = preprocess_image(image_path)
if preprocessed_image is not None:
    height, width = preprocessed_image.shape

    # Create a mask: left half white (255), right half black (0)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[:, :width // 2] = 255

    and_result = apply_logical_and(preprocessed_image, mask)
    or_result = apply_logical_or(preprocessed_image, mask)
    not_result = apply_logical_not(preprocessed_image)
    xor_result = apply_logical_xor(preprocessed_image, mask)

    # Create collage: AND | OR | NOT | XOR
    collage = np.zeros((height, width * 4), dtype=np.uint8)
    collage[:, :width] = and_result
    collage[:, width:width * 2] = or_result
    collage[:, width * 2:width * 3] = not_result
    collage[:, width * 3:] = xor_result

    collage_pil = Image.fromarray(collage)
    collage_pil.save('20701014_Q19/20701014_Q19_output.jpg')
