from PIL import Image
import numpy as np
import math, random

def separate_rgb_channels(image_path):
    try:
        image = Image.open(image_path)
        image_array = np.array(image)
        
        if len(image_array.shape) != 3 or image_array.shape[2] != 3:
            raise ValueError("Input image must be RGB (3 channels)")
        
        height, width = image_array.shape[:2]
        
        red_channel = image_array[:, :, 0]   
        green_channel = image_array[:, :, 1] 
        blue_channel = image_array[:, :, 2]  
        
        red_image = Image.fromarray(red_channel)
        green_image = Image.fromarray(green_channel)
        blue_image = Image.fromarray(blue_channel)
        
        red_image.save('20701014_Q12/20701014_Q12_red_channel.jpg')
        green_image.save('20701014_Q12/20701014_Q12_green_channel.jpg')
        blue_image.save('20701014_Q12/20701014_Q12_blue_channel.jpg')
        
        collage_width = width * 2
        collage_height = height * 2
        collage = Image.new('RGB', (collage_width, collage_height), (255, 255, 255)) 
        
        collage.paste(image, (0, 0))
        red_rgb = Image.fromarray(np.stack([red_channel, red_channel, red_channel], axis=2))
        green_rgb = Image.fromarray(np.stack([green_channel, green_channel, green_channel], axis=2))
        blue_rgb = Image.fromarray(np.stack([blue_channel, blue_channel, blue_channel], axis=2))
        
        collage.paste(red_rgb, (width, 0))        
        collage.paste(green_rgb, (0, height))     
        collage.paste(blue_rgb, (width, height))  
        
        collage.save('20701014_Q12/20701014_Q12_output.jpg')
        
    except Exception as e:
        print(f"Error processing image: {e}")

image_path = '20701014_Q12/20701014_Q12_input.jpg'
separate_rgb_channels(image_path)