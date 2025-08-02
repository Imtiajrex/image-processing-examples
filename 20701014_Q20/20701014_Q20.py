from PIL import Image
import numpy as np
import math, random

def apply_kmeans_segmentation(image, k=3, max_iterations=10):
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Image must be RGB (3 channels)")
    
    height, width = image.shape[:2]
    
    pixels = image.reshape(-1, 3).astype(np.float32)
    num_pixels = pixels.shape[0]
    
    indices = random.sample(range(num_pixels), k)
    centroids = pixels[indices]
    
    # K-means iteration
    for _ in range(max_iterations):
        # Assign each pixel to the nearest centroid
        distances = np.zeros((num_pixels, k))
        for i in range(num_pixels):
            for j in range(k):
                r_diff = pixels[i, 0] - centroids[j, 0]
                g_diff = pixels[i, 1] - centroids[j, 1]
                b_diff = pixels[i, 2] - centroids[j, 2]
                distances[i, j] = math.sqrt(r_diff * r_diff + g_diff * g_diff + b_diff * b_diff)
        labels = np.argmin(distances, axis=1)
        
        # update centroids
        new_centroids = np.zeros((k, 3))
        counts = np.zeros(k, dtype=np.int32)
        for i in range(num_pixels):
            cluster = labels[i]
            new_centroids[cluster] += pixels[i]
            counts[cluster] += 1
        for j in range(k):
            if counts[j] > 0:
                new_centroids[j] /= counts[j]
        
        # check convergence
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    
    # assignment to final clusters
    segmented_pixels = np.zeros((num_pixels, 3), dtype=np.uint8)
    for i in range(num_pixels):
        cluster = labels[i]
        segmented_pixels[i] = centroids[cluster].astype(np.uint8)
    
    segmented_image = segmented_pixels.reshape(height, width, 3)
    
    return segmented_image

image_path = '20701014_Q20/20701014_Q20_input.jpg'
image = Image.open(image_path)
image_array = np.array(image)
if image_array is not None:
    segmented_image = apply_kmeans_segmentation(image_array, k=3, max_iterations=10)
    segmented_image_pil = Image.fromarray(segmented_image)
    segmented_image_pil.save('20701014_Q20/20701014_Q20_output.jpg')