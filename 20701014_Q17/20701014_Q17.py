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

def apply_gamma_correction(image, gamma):
    if len(image.shape) != 2:
        raise ValueError("Image must be grayscale (2D)")

    height, width = image.shape
    corrected = np.zeros((height, width), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            # Normalize to [0, 1], apply gamma, scale back to [0, 255]
            normalized = image[i, j] / 255.0
            corrected[i, j] = int((normalized ** gamma) * 255)

    return corrected

image_path = '20701014_Q17/20701014_Q17_input.jpg'
preprocessed_image = preprocess_image(image_path)
if preprocessed_image is not None:
    # gamma < 1 brightens the image, gamma > 1 darkens it
    brightened = apply_gamma_correction(preprocessed_image, gamma=0.5)
    darkened = apply_gamma_correction(preprocessed_image, gamma=2.0)

    # Create collage: original | brightened (gamma=0.5) | darkened (gamma=2.0)
    height, width = preprocessed_image.shape
    collage = np.zeros((height, width * 3), dtype=np.uint8)
    collage[:, :width] = preprocessed_image
    collage[:, width:width * 2] = brightened
    collage[:, width * 2:] = darkened

    collage_pil = Image.fromarray(collage)
    collage_pil.save('20701014_Q17/20701014_Q17_output.jpg')
