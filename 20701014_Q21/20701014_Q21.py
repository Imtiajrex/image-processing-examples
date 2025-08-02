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

def median_filter(image, kernel_size=3):
    height, width = image.shape
    padded = np.pad(image, ((kernel_size//2, kernel_size//2), (kernel_size//2, kernel_size//2)), mode='constant')
    filtered = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            window = padded[i:i+kernel_size, j:j+kernel_size].flatten()
            filtered[i, j] = np.median(window)
    return filtered

def remove_small_objects(image, min_size=50):
    height, width = image.shape
    binary_image = np.zeros((height, width), dtype=np.uint8)
    threshold = 128
    for i in range(height):
        for j in range(width):
            binary_image[i, j] = 255 if image[i, j] >= threshold else 0
    
    labels = np.zeros((height, width), dtype=np.int32)
    object_sizes = {}
    label = 0
    for i in range(height):
        for j in range(width):
            if binary_image[i, j] == 255 and labels[i, j] == 0:
                label += 1
                size = 0
                stack = [(i, j)]
                while stack:
                    x, y = stack.pop()
                    if 0 <= x < height and 0 <= y < width and binary_image[x, y] == 255 and labels[x, y] == 0:
                        labels[x, y] = label
                        size += 1
                        stack.append((x-1, y))
                        stack.append((x+1, y))
                        stack.append((x, y-1))
                        stack.append((x, y+1))
                object_sizes[label] = size
    
    cleaned_image = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            if labels[i, j] > 0 and object_sizes.get(labels[i, j], 0) >= min_size:
                cleaned_image[i, j] = 255
    return cleaned_image

def histogram_equalization(image):
    height, width = image.shape
    hist = np.zeros(256, dtype=np.int32)
    for i in range(height):
        for j in range(width):
            hist[image[i, j]] += 1
    
    cdf = np.zeros(256, dtype=np.float32)
    cdf[0] = hist[0]
    for i in range(1, 256):
        cdf[i] = cdf[i-1] + hist[i]
    
    cdf_min = cdf[np.nonzero(cdf)[0][0]] if np.any(cdf) else 0
    cdf_max = cdf[-1]
    equalized = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            if cdf_max > cdf_min:
                equalized[i, j] = int(((cdf[image[i, j]] - cdf_min) * 255) / (cdf_max - cdf_min) + 0.5)
    return equalized

def apply_watershed_segmentation(image):
    if len(image.shape) != 2:
        raise ValueError("Image must be grayscale (2D)")
    
    # Preprocessing
    image = median_filter(image, kernel_size=3)
    Image.fromarray(image).save('20701014_Q21/20701014_Q21_reducedNoise.jpg')
    image = remove_small_objects(image, min_size=50)
    Image.fromarray(image).save('20701014_Q21/20701014_Q21_removedSmallObjects.jpg')
    image = histogram_equalization(image)
    Image.fromarray(image).save('20701014_Q21/20701014_Q21_histogramEqualized.jpg')
    
    height, width = image.shape
    
    # Compute gradient magnitude (simple difference)
    gradient = np.zeros((height, width), dtype=np.uint8)
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            dx = image[i, j + 1] - image[i, j - 1]
            dy = image[i + 1, j] - image[i - 1, j]
            gradient[i, j] = min(255, int(math.sqrt(dx * dx + dy * dy) + 0.5))
    
    # Detect markers (local minima)
    markers = np.zeros((height, width), dtype=np.int32)
    current_marker = 1
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            is_minima = True
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    if image[i + di, j + dj] < image[i, j]:
                        is_minima = False
                        break
                if not is_minima:
                    break
            if is_minima and image[i, j] < 150:  # Adjusted threshold
                markers[i, j] = current_marker
                current_marker += 1
    
    # Watershed algorithm
    segmented_image = np.zeros((height, width), dtype=np.uint8)
    labels = np.zeros((height, width), dtype=np.int32)
    stack = []
    for i in range(height):
        for j in range(width):
            if markers[i, j] > 0:
                labels[i, j] = markers[i, j]
                stack.append((i, j))
    
    while stack:
        x, y = stack.pop()
        for di in range(-1, 2):
            for dj in range(-1, 2):
                ni, nj = x + di, y + dj
                if (0 <= ni < height and 0 <= nj < width and
                    labels[ni, nj] == 0 and gradient[ni, nj] < gradient[x, y]):
                    labels[ni, nj] = labels[x, y]
                    stack.append((ni, nj))
    
    # Assign unique colors to segments and add watershed lines
    max_label = np.max(labels)
    color_map = np.zeros((max_label + 1, 3), dtype=np.uint8)
    for i in range(1, max_label + 1):
        color_map[i] = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
    for i in range(height):
        for j in range(width):
            if labels[i, j] > 0:
                segmented_image[i, j] = color_map[labels[i, j], 0]  # Use R for grayscale display
            else:
                segmented_image[i, j] = 0  # Black for watershed lines
    
    return segmented_image

image_path = '20701014_Q21/20701014_Q21_input.jpg'
preprocessed_image = preprocess_image(image_path)
if preprocessed_image is not None:
    segmented_image = apply_watershed_segmentation(preprocessed_image)
    segmented_image_pil = Image.fromarray(segmented_image)  
    segmented_image_pil.save('20701014_Q21/20701014_Q21_output.jpg')