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

def apply_bilinear_interpolation(image, scale_x, scale_y):
    if len(image.shape) != 2:
        raise ValueError("Image must be grayscale (2D)")

    src_height, src_width = image.shape
    dst_height = int(src_height * scale_y)
    dst_width = int(src_width * scale_x)

    output = np.zeros((dst_height, dst_width), dtype=np.uint8)

    for i in range(dst_height):
        for j in range(dst_width):
            # Map destination pixel back to source image coordinates
            src_x = j / scale_x
            src_y = i / scale_y

            # Four surrounding pixel coordinates
            x0 = int(src_x)
            y0 = int(src_y)
            x1 = min(x0 + 1, src_width - 1)
            y1 = min(y0 + 1, src_height - 1)

            # Fractional parts
            a = src_x - x0  # horizontal fraction
            b = src_y - y0  # vertical fraction

            # Bilinear interpolation formula
            interpolated = (
                (1 - a) * (1 - b) * image[y0, x0] +
                a * (1 - b) * image[y0, x1] +
                (1 - a) * b * image[y1, x0] +
                a * b * image[y1, x1]
            )
            output[i, j] = int(interpolated)

    return output

image_path = '20701014_Q16/20701014_Q16_input.jpg'
preprocessed_image = preprocess_image(image_path)
if preprocessed_image is not None:
    output_image = apply_bilinear_interpolation(preprocessed_image, scale_x=1.5, scale_y=1.5)
    output_image_pil = Image.fromarray(output_image)
    output_image_pil.save('20701014_Q16/20701014_Q16_output.jpg')
