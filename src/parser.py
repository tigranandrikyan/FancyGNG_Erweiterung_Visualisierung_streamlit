from PIL import Image
import numpy as np
import glob
import constants
import os
import shutil
from datetime import datetime

# Data preparation

# Reads images, converts them, normalizes, and stores their sizes
def parse(file_list):  # Defines the parse function which takes a list of image files (file_list) as input
    data_list = []  # Stores image data as normalized arrays
    size_images = []  # Stores image sizes
    for image_file in file_list:  # Loop over each image in file_list
        try:  # Attempt to execute the following block to catch errors
            image = Image.open(image_file)  # Open the image file
            if image.mode != 'RGB':  # Check if the image is not already in RGB mode
                image = image.convert('RGB')  # Convert to RGB if necessary
            size_images.append(image.size)  # Store the image size (width, height) in size_images
            data_array = np.vstack(np.asarray(image))  # Convert image to NumPy array and stack rows vertically
            data_list.append(data_array / constants.MAX_COLOR_VALUE)  # Normalize and store in data_list
        except Exception as e:  # Catch errors
            print(f"Error processing image {image_file}: {str(e)}")
    return data_list, size_images  # Return list of normalized image data and sizes

# Creates a list of all image files in a directory
def generate_file_list(image_in_path='./data'):  # Default path './data'
    # glob: searches filesystem for files; '/*' -> all files in folder; FILE_TYPE: file extension from constants.py
    file_list = glob.glob(image_in_path + '/*' + constants.FILE_TYPE)
    return file_list  # Return the list of found image files

# Extracts image names without extension
def get_image_names(file_list):  # Receives a list of file paths (file_list)
    name_list = []  # Store image names
    for file_path in file_list:  # Loop over each file path
        image = os.path.basename(file_path)  # Extract filename from path
        image_name = os.path.splitext(image)[0]  # Remove file extension
        name_list.append(image_name)  # Append the name to name_list
    return name_list  # Return list of image names



# Saves augmented images to a specified directory
def save_data(aug_data, name, size_images, aug_count, data_index, path='./out_data'):  
    # path='./out_data': destination for saving image (default: './out_data')
    aug_image = aug_data * constants.MAX_COLOR_VALUE  # Scale normalized pixel values back to original range

    # Reshape the 1D array back to original image size and type uint8 (required for images)
    aug_image = aug_image.reshape((size_images[data_index][1], size_images[data_index][0], 3)).astype(np.uint8)  # [1]: height, [0]: width, 3: RGB

    image = Image.fromarray(aug_image)  # Create a PIL image from the NumPy array
    final_image = image.convert("RGB")  # Convert image to RGB mode again (if needed)

    # Save image with numbered suffix (name_1, name_2, ...) and FILE_TYPE from constants.py
    # Platform-independent solution for Windows/Mac/Linux:
    save_path = os.path.join(path, name + "_" + str(aug_count + 1) + constants.FILE_TYPE)
    final_image.save(save_path)
