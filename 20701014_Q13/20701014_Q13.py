from PIL import Image
import numpy as np
import math, random

def rgb_to_hsv(rgb_array):
    height, width = rgb_array.shape[:2]
    hsv_array = np.zeros((height, width, 3), dtype=np.float32)
    
    for i in range(height):
        for j in range(width):
            # normalize RGB to [0, 1]
            r = rgb_array[i, j, 0] / 255.0
            g = rgb_array[i, j, 1] / 255.0
            b = rgb_array[i, j, 2] / 255.0
            
            # Value (V)
            v = max(r, g, b)
            
            # Saturation (S)
            min_val = min(r, g, b)
            s = 0 if v == 0 else (v - min_val) / v
            
            # Hue (H)
            h = 0
            if v != min_val:
                if v == r:
                    h = 60 * ((g - b) / (v - min_val)) % 360
                elif v == g:
                    h = 60 * ((b - r) / (v - min_val) + 2)
                elif v == b:
                    h = 60 * ((r - g) / (v - min_val) + 4)
            
            h = (h / 360.0) * 255.0  # Hue: 0-360 -> 0-255
            s = s * 255.0            # Saturation: 0-1 -> 0-255
            v = v * 255.0            # Value: 0-1 -> 0-255
            
            hsv_array[i, j] = [h, s, v]
    
    return hsv_array.astype(np.uint8)

def convert_and_collage_hsv_manual(image_path):
    try:
        rgb_image = Image.open(image_path)
        image_array = np.array(rgb_image)
        
        if len(image_array.shape) != 3 or image_array.shape[2] != 3:
            raise ValueError("Input image must be RGB (3 channels)")
        
        height, width = image_array.shape[:2]
        
        hsv_array = rgb_to_hsv(image_array)
        
        hue_channel = hsv_array[:, :, 0]
        sat_channel = hsv_array[:, :, 1]
        val_channel = hsv_array[:, :, 2]

        hue_image = Image.fromarray(hue_channel)
        sat_image = Image.fromarray(sat_channel)
        val_image = Image.fromarray(val_channel)
        
        hue_image.save('20701014_Q13/20701014_Q13_hue_channel.jpg')
        sat_image.save('20701014_Q13/20701014_Q13_saturation_channel.jpg')
        val_image.save('20701014_Q13/20701014_Q13_value_channel.jpg')
        
        collage_width = width * 2
        collage_height = height * 2
        collage = Image.new('RGB', (collage_width, collage_height), (255, 255, 255))
        
        collage.paste(rgb_image, (0, 0))
        
        hue_rgb = Image.fromarray(np.stack([hue_channel, hue_channel, hue_channel], axis=2))
        sat_rgb = Image.fromarray(np.stack([sat_channel, sat_channel, sat_channel], axis=2))
        val_rgb = Image.fromarray(np.stack([val_channel, val_channel, val_channel], axis=2))
        
        collage.paste(hue_rgb, (width, 0))        # Top-right: Hue
        collage.paste(sat_rgb, (0, height))       # Bottom-left: Saturation
        collage.paste(val_rgb, (width, height))   # Bottom-right: Value
        
        collage.save('20701014_Q13/20701014_Q13_output.jpg')
        
    except Exception as e:
        print(f"Error processing image: {e}")

image_path = '20701014_Q13/20701014_Q13_input.jpg'
convert_and_collage_hsv_manual(image_path)