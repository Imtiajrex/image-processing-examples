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

def detect_sobel_edges(image, kernel_size=3, vertical=True):
    if len(image.shape) != 2:
        raise ValueError("Image must be grayscale (2D)")
    height, width = image.shape

    if vertical:
        kernel = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=np.float32) # horizontal sobel kernel
    else:
        kernel = np.array([[1, 2, 1],
                           [0, 0, 0],
                           [-1, -2, -1]], dtype=np.float32) # vertical sobel kernel

    # padding the image to handle borders
    pad_amount = kernel_size // 2
    padded_image = np.pad(image, pad_amount, mode='constant', constant_values=0)

    # applying the sobel filter
    filtered_image = np.zeros_like(image, dtype=np.float32)
    for i in range(height):
        for j in range(width):
            neighbors = padded_image[i:i + kernel_size, j:j + kernel_size]
            filtered_image[i, j] = np.sum(neighbors * kernel)
    output = np.clip(filtered_image, 0, 255).astype(np.uint8)
    return output

# combine edges from both vertical and horizontal sobel kernels
def combine_edges(vertical_edges, horizontal_edges):
    Gx = vertical_edges.astype(np.float32)
    Gy = horizontal_edges.astype(np.float32)
    G_magnitude = np.sqrt(Gx**2 + Gy**2)
    min_val = np.min(G_magnitude)
    max_val = np.max(G_magnitude)
    if max_val - min_val > 0:
        G_magnitude = (G_magnitude - min_val) / (max_val - min_val) * 255
    else:
        G_magnitude = np.zeros_like(G_magnitude)
    return G_magnitude.clip(0, 255).astype(np.uint8)


preprocessed_image = preprocess_image('Sobel_Operation/Sobel_Operation_input.jpg')
vertical_edges = detect_sobel_edges(image = preprocessed_image, vertical=True)
horizontal_edges = detect_sobel_edges(image= preprocessed_image, vertical=False)

vertical_edges_pil = Image.fromarray(np.uint8(vertical_edges))
horizontal_edges_pil = Image.fromarray(np.uint8(horizontal_edges))
horizontal_edges_pil.save('Sobel_Operation/Sobel_Operation_horizontal_output.jpg')
vertical_edges_pil.save('Sobel_Operation/Sobel_Operation_vertical_output.jpg')

# create collage of both edge images
collage_width = vertical_edges.shape[1] + horizontal_edges.shape[1]
collage_height = max(vertical_edges.shape[0], horizontal_edges.shape[0])
collage = np.zeros((collage_height, collage_width), dtype=np.uint8)
collage[:vertical_edges.shape[0], :vertical_edges.shape[1]] = vertical_edges
collage[:horizontal_edges.shape[0], vertical_edges.shape[1]:] = horizontal_edges

collage_pil = Image.fromarray(np.uint8(collage))
collage_pil.save('Sobel_Operation/Sobel_Operation_output.jpg')