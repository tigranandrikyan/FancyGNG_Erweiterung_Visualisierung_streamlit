import constants
import parser
import fancy_pca as FP

# Start Fancy PCA

file_list = parser.generate_file_list()  # Generate a list of files to process
data, size_images = parser.parse(file_list)  # Parse the files and extract image data and their sizes
data_names = parser.get_image_names(file_list)  # Retrieve the names of the images from the file list



fancy_pca_transform = FP.FancyPCA()  # Create an instance of the FancyPCA transformation

# Iterate over all files in the list
for data_index in range(len(file_list)):

    # Repeat the augmentation for the number of times specified by 'AUG_COUNT' in constants.py
    for aug_count in range(constants.AUG_COUNT):
        print(f"Fancy_PCA: {str(data_names[data_index])}, {aug_count + 1}/{constants.AUG_COUNT}")  # Debug: print image name and augmentation progress
        fancy_pca_images = fancy_pca_transform.fancy_pca(data[data_index])  # Apply the FancyPCA transformation to the current image

        # Save the transformed image with the given parameters in the output folder
        parser.save_data(
            fancy_pca_images,
            data_names[data_index],
            size_images,
            aug_count,
            data_index,
            path='./out_fancy_pca'
        )
