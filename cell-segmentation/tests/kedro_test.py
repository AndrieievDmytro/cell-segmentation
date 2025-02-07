# from PIL import Image
# from pathlib import Path
# from datetime import time
# from pathlib import Path


# input_folder = Path("E:/diploma_proj_latest/cell-segmentation/.data/01_raw/livecell_test")
# for file in input_folder.glob("*.tif"):
#     try:
#         with Image.open(file) as img:
#             print(f"Successfully opened {file.name}")
#     except Exception as e:
#         print(f"Error opening {file.name}: {e}")


# import os

# def check_user():
#     print("Checking user access...")
#     print(f"Current user: {os.getlogin()}")

# if __name__ == "__main__":
#     check_user()


# from pathlib import Path

# def test_folder_access():
#     path = Path("E:/diploma_proj_latest/cell-segmentation/.data/01_raw/livecell_test")
#     try:
#         print(f"Checking folder access for: {path}")
#         if path.exists() and path.is_dir():
#             print(f"Folder exists and is a directory: {path}")
#             for file in path.glob("*.tif"):
#                 print(f"File found: {file}")
#         else:
#             print(f"Folder not accessible: {path}")
#     except PermissionError as e:
#         print(f"Permission error: {e}")
#     except Exception as e:
#         print(f"Unexpected error: {e}")

# if __name__ == "__main__":
#     test_folder_access()


# import os
# from pathlib import Path

# def compare_directories(dir1, dir2):
#     dir1_path = Path(dir1)
#     dir2_path = Path(dir2)

#     print(f"Files in {dir1_path}: {list(dir1_path.iterdir())}")
#     print(f"Files in {dir2_path}: {list(dir2_path.iterdir())}")

#     # Get list of files in each directory (case-insensitive extensions)
#     dir1_files = {file.name: file for file in dir1_path.glob("*.tif")}
#     dir1_files.update({file.name: file for file in dir1_path.glob("*.TIF")})

#     dir2_files = {file.name: file for file in dir2_path.glob("*.tif")}
#     dir2_files.update({file.name: file for file in dir2_path.glob("*.TIF")})

#     # Compare files
#     missing_in_dir2 = dir1_files.keys() - dir2_files.keys()
#     missing_in_dir1 = dir2_files.keys() - dir1_files.keys()

#     common_files = dir1_files.keys() & dir2_files.keys()
#     size_mismatched_files = []

#     for file_name in common_files:
#         if dir1_files[file_name].stat().st_size != dir2_files[file_name].stat().st_size:
#             size_mismatched_files.append(file_name)

#     # Count total files in each directory
#     total_files_dir1 = len(dir1_files)
#     total_files_dir2 = len(dir2_files)

#     # Print results
#     print(f"\nDirectory: {dir1}")
#     print(f"Total number of files: {total_files_dir1}")

#     print(f"\nDirectory: {dir2}")
#     print(f"Total number of files: {total_files_dir2}")

#     print(f"\nFiles in {dir1} but missing in {dir2} ({len(missing_in_dir2)}):")
#     for file_name in missing_in_dir2:
#         print(f"  - {file_name}")

#     print(f"\nFiles in {dir2} but missing in {dir1} ({len(missing_in_dir1)}):")
#     for file_name in missing_in_dir1:
#         print(f"  - {file_name}")

#     print(f"\nFiles present in both but with size mismatch ({len(size_mismatched_files)}):")
#     for file_name in size_mismatched_files:
#         dir1_size = dir1_files[file_name].stat().st_size
#         dir2_size = dir2_files[file_name].stat().st_size
#         print(f"  - {file_name}: {dir1} size={dir1_size}, {dir2} size={dir2_size}")

# if __name__ == "__main__":
#     raw_dir = "E:/diploma_proj_latest/cell-segmentation/.data/01_raw"
#     intermediate_dir = "E:/diploma_proj_latest/cell-segmentation/.data/02_intermediate"

#     compare_directories(raw_dir, intermediate_dir)

# import tensorflow as tf
# print("TensorFlow version:", tf.__version__)

# # Check GPU availability
# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     print(f"GPUs Available: {len(gpus)}")
#     for gpu in gpus:
#         print(f"GPU: {gpu}")
# else:
#     print("No GPUs detected.")

# # Test GPU computation
# if gpus:
#     with tf.device('/GPU:0'):
#         a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
#         b = tf.constant([[1.0, 0.0], [0.0, 1.0]])
#         c = tf.matmul(a, b)
#         print("GPU computation result:\n", c)

# import tensorflow as tf

# print("TensorFlow version:", tf.__version__)
# print("CUDA version TensorFlow was built with:", tf.sysconfig.get_build_info()['cuda_version'])
# print("cuDNN version TensorFlow was built with:", tf.sysconfig.get_build_info()['cudnn_version'])


# import torch
# print(torch.__version__)  # PyTorch version
# print(torch.cuda.is_available())  # Check if CUDA is available
# print(torch.version.cuda)  # CUDA version bundled with PyTorch
# print(torch.cuda.get_device_name(0))  # Your GPU name


# Normalizatuion check 

# from PIL import Image
# import numpy as np

# def check_normalization(image_path):
#     with Image.open(image_path) as img:
#         arr = np.array(img, dtype=np.float32)

#     print(f"Image shape: {arr.shape}")
#     print(f"Min pixel value: {arr.min()}")
#     print(f"Max pixel value: {arr.max()}")
#     print(f"Mean pixel value: {arr.mean()}")
#     print(f"Standard deviation: {arr.std()}")

# # Example usage
# check_normalization("E:\diploma_proj_latest\cell-segmentation\.data/03_processed/normalized\livecell_test\A172\A172_Phase_C7_1_00d00h00m_1.tif")

# Check correspondance of all train images in mask and normalized 


# def check_image_mask_correspondence(normalized_train_folder, masks_folder):
#     """
#     Check that all images in the normalized train folder have corresponding masks
#     and vice versa.
#     """
#     normalized_train_path = Path(normalized_train_folder)
#     masks_path = Path(masks_folder)
    
#     mismatched_files = []

#     # Check each class folder
#     for class_folder in normalized_train_path.iterdir():
#         if class_folder.is_dir():
#             class_name = class_folder.name
#             class_masks_folder = masks_path / class_name
            
#             # Check if the class folder exists in the masks folder
#             if not class_masks_folder.exists():
#                 print(f"Class folder {class_name} is missing in the masks directory.")
#                 mismatched_files.append((class_name, "Missing mask class folder"))
#                 continue
            
#             # List files in both folders
#             train_files = {file.stem for file in class_folder.glob("*.npy")}
#             mask_files = {file.stem for file in class_masks_folder.glob("*.npy")}
            
#             # Find mismatches
#             missing_masks = train_files - mask_files
#             missing_images = mask_files - train_files
            
#             # Report mismatches
#             if missing_masks:
#                 print(f"Missing masks for class '{class_name}': {missing_masks}")
#                 mismatched_files.extend([(class_name, f"Missing mask: {name}.npy") for name in missing_masks])
            
#             if missing_images:
#                 print(f"Missing images for class '{class_name}': {missing_images}")
#                 mismatched_files.extend([(class_name, f"Missing image: {name}.npy") for name in missing_images])
    
#     if not mismatched_files:
#         print("All image-mask correspondences are correct.")
#     else:
#         print("Some mismatches were found.")
    
#     return mismatched_files

# # Example usage
# normalized_train_folder = "E:/diploma_proj_latest/cell-segmentation/.data/03_processed/normalized/livecell_train"
# masks_folder = "E:/diploma_proj_latest/cell-segmentation/.data/03_processed/masks/livecell_train"

# mismatches = check_image_mask_correspondence(normalized_train_folder, masks_folder)


# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# from skimage.morphology import remove_small_objects, disk
# import tifffile as tiff
# import os

# def load_and_normalize_image(image_path):
#     """Load a TIFF image and normalize it to 0-255."""
#     if not os.path.exists(image_path):
#         raise FileNotFoundError(f"Image not found at {image_path}")
#     image = tiff.imread(image_path)
#     image_norm = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
#     return image_norm

# def enhance_contrast(image):
#     """Enhance image contrast using CLAHE."""
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     enhanced = clahe.apply(image)
#     return enhanced

# def plot_histogram(image, title="Histogram"):
#     """Plot the intensity histogram of the image."""
#     plt.figure(figsize=(8, 4))
#     plt.hist(image.ravel(), bins=256, range=(0, 256))
#     plt.title(title)
#     plt.xlabel("Pixel Intensity")
#     plt.ylabel("Frequency")
#     plt.show()

# def threshold_and_morphology(enhanced_image, threshold_val=40, use_adaptive=False, use_otsu=False,
#                             invert_threshold=True, min_obj_size=50):
#     """
#     Apply thresholding to create a binary mask and perform minimal cleanup.
    
#     Parameters:
#     - use_adaptive: Use adaptive thresholding if True.
#     - use_otsu: Use Otsu's method for automatic threshold selection.
#     - invert_threshold: Invert the binary mask (useful for dark cells on a bright background).
#     - min_obj_size: Minimum size for objects to keep (in pixels).
    
#     This function will display a single debug plot of the binary mask immediately after thresholding.
#     """
#     # Choose the thresholding method:
#     if use_adaptive:
#         if invert_threshold:
#             binary_mask = cv2.adaptiveThreshold(enhanced_image, 255,
#                                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                                 cv2.THRESH_BINARY_INV, 11, 2)
#         else:
#             binary_mask = cv2.adaptiveThreshold(enhanced_image, 255,
#                                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                                 cv2.THRESH_BINARY, 11, 2)
#     elif use_otsu:
#         if invert_threshold:
#             ret, binary_mask = cv2.threshold(enhanced_image, 0, 255,
#                                             cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#         else:
#             ret, binary_mask = cv2.threshold(enhanced_image, 0, 255,
#                                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     else:
#         if invert_threshold:
#             ret, binary_mask = cv2.threshold(enhanced_image, threshold_val, 255, cv2.THRESH_BINARY_INV)
#         else:
#             ret, binary_mask = cv2.threshold(enhanced_image, threshold_val, 255, cv2.THRESH_BINARY)
    
#     # Convert to boolean mask:
#     binary_mask = binary_mask > 0

#     # --- Debug: Display only the binary mask immediately after thresholding ---
#     plt.figure(figsize=(6, 6))
#     plt.imshow(binary_mask, cmap="gray")
#     plt.title("Binary Mask After Thresholding")
#     plt.axis("off")
#     plt.show()

#     # # Optionally remove small objects (minimal cleanup)
#     binary_mask_clean = remove_small_objects(binary_mask, min_size=min_obj_size)
    
#     return binary_mask_clean

# def visualize_results_binary(original, binary_mask):
#     """Display the original image and the binary mask side-by-side."""
#     fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
#     axes[0].imshow(original, cmap="gray")
#     axes[0].set_title("Original Image")
#     axes[0].axis("off")
    
#     axes[1].imshow(binary_mask, cmap="gray")
#     axes[1].set_title("Binary Mask After Thresholding")
#     axes[1].axis("off")
    
#     plt.tight_layout()
#     plt.show()

# def main():
#     # Specify the path to your TIFF image (512x515 grayscale)
#     # Uncomment the image you wish to use:
#     image_path = r"E:\diploma_proj_latest\cell-segmentation\.data\02_intermediate\livecell_train\A172\A172_Phase_A7_1_00d00h00m_2.tif"
#     # image_path = r"E:\diploma_proj_latest\cell-segmentation\.data\02_intermediate\livecell_train\BT474\BT474_Phase_B3_2_04d00h00m_4.tif"
#     # image_path = r"E:\diploma_proj_latest\cell-segmentation\.data\02_intermediate\livecell_train\BV-2\BV2_Phase_B4_1_00d00h00m_1.tif"
#     # image_path = r"E:\diploma_proj_latest\cell-segmentation\.data\02_intermediate\livecell_train\Huh7\Huh7_Phase_A10_1_00d00h00m_1.tif"
#     # image_path = r"E:\diploma_proj_latest\cell-segmentation\.data\02_intermediate\livecell_train\MCF7\MCF7_Phase_E4_1_00d00h00m_2.tif"
#     # image_path = r"E:\diploma_proj_latest\cell-segmentation\.data\02_intermediate\livecell_train\SH-SHY5Y\SHSY5Y_Phase_B10_1_00d00h00m_1.tif"
#     # image_path = r"E:\diploma_proj_latest\cell-segmentation\.data\02_intermediate\livecell_train\SK-OV-3\SKOV3_Phase_G4_1_00d00h00m_1.tif"
#     # image_path = r"E:\diploma_proj_latest\cell-segmentation\.data\02_intermediate\livecell_train\SkBr3\SkBr3_Phase_E3_1_00d00h00m_1.tif"
    
#     image_norm = load_and_normalize_image(image_path)
#     enhanced_image = enhance_contrast(image_norm)
    
#     print("Enhanced image min:", np.min(enhanced_image))
#     print("Enhanced image max:", np.max(enhanced_image))
#     plot_histogram(enhanced_image, "Histogram of Enhanced Image")
    
#     # --- Set thresholding parameters ---
#     use_adaptive = False    # Change to True if illumination is uneven
#     use_otsu = True         # Use Otsu's method to auto-select threshold
#     invert_threshold = True # Invert thresholding since cells are dark
#     threshold_val = 40      # Only used if not using Otsu
#     min_obj_size = 5       # Lowered to preserve smaller cells
    
#     # Apply thresholding and minimal cleanup:
#     binary_mask = threshold_and_morphology(enhanced_image, threshold_val, use_adaptive,
#                                         use_otsu, invert_threshold, min_obj_size)
    
#     # Visualize only the original image and the binary mask.

#     binary_mask = (binary_mask / binary_mask.max() * 255).astype(np.uint8)
#     visualize_results_binary(image_norm, binary_mask)
#     tiff.imwrite(r"tets_pic.tif", binary_mask)

# if __name__ == "__main__":
#     main()


# import tifffile as tiff
# import numpy as np
# import matplotlib.pyplot as plt
# import cv2
# from skimage.morphology import remove_small_objects

# # File path to analyze
# # file_path = r"E:\diploma_proj_latest\cell-segmentation\.data\03_processed\masks\livecell_test\test\A172\A172_Phase_C7_1_00d00h00m_1.tif"
# file_path = r"E:\diploma_proj_latest\cell-segmentation\.data\03_processed\masks\livecell_test\test\A172\A172_Phase_C7_1_00d00h00m_1.tif"

# # Function to enhance contrast (CLAHE)
# def enhance_contrast(image):
#     """Enhance image contrast using CLAHE."""
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     return clahe.apply(image.astype(np.uint8))

# try:
#     # Load the TIFF image
#     image = tiff.imread(file_path)

#     # Extract image properties
#     image_shape = image.shape
#     image_dtype = image.dtype
#     unique_values = np.unique(image)

#     # Display key properties
#     print(f"Image Shape: {image_shape}")
#     print(f"Data Type: {image_dtype}")
#     print(f"Min Pixel Value: {image.min()}, Max Pixel Value: {image.max()}")
#     print(f"Unique Values (first 20): {unique_values[:20]}")  # Show only first 20 unique values

#     # Check if the image contains only grayscale (0-255) or discrete classes (0,1,2,...)
#     is_grayscale = (image.max() <= 255) and (len(unique_values) > 20)  # Many unique values indicate grayscale
#     is_class_mask = (len(unique_values) <= 10)  # Few unique values indicate segmentation labels

#     print(f"Is the image grayscale? {'Yes' if is_grayscale else 'No'}")
#     print(f"Is the image a class segmentation mask? {'Yes' if is_class_mask else 'No'}")

#     # ---- Debugging Visualizations ----

#     # Plot histogram of pixel values
#     plt.figure(figsize=(8, 4))
#     plt.hist(image.ravel(), bins=256, range=(0, image.max()), color="blue", alpha=0.7)
#     plt.title("Pixel Value Distribution")
#     plt.xlabel("Pixel Intensity")
#     plt.ylabel("Frequency")
#     plt.show()

#     # Enhance contrast (for better visibility of cells)
#     enhanced_image = enhance_contrast(image)

#     # Apply thresholding (dynamic or fixed)
#     threshold_value = 1  # Adjust as necessary
#     _, binary_mask = cv2.threshold(enhanced_image, threshold_value, 1, cv2.THRESH_BINARY)

#     # Remove small objects (cleanup)
#     binary_mask_clean = remove_small_objects(binary_mask > 0, min_size=50)

#     # Overlay mask on the original image
#     plt.figure(figsize=(10, 10))
#     plt.imshow(image, cmap="gray", alpha=0.8)  # Original image
#     plt.imshow(binary_mask_clean, cmap="jet", alpha=0.5)  # Overlay mask
#     plt.title("Overlay: Image + Mask")
#     plt.axis("off")
#     plt.show()

#     # Save a visualization-friendly mask
#     visual_mask = (binary_mask_clean * 255).astype(np.uint8)
#     plt.imshow(visual_mask, cmap="gray")
#     plt.title("Binary Mask for Visualization")
#     plt.axis("off")
#     plt.show()

# except ValueError as e:
#     print(f"Error loading TIFF file: {e}")
#     print("This may be due to LZW compression. Try saving the file as an uncompressed TIFF or installing 'imagecodecs'.")


# import tifffile as tiff
# import numpy as np
# import matplotlib.pyplot as plt

# # File path to the binary mask
# file_path = r"E:\diploma_proj_latest\cell-segmentation\.data\03_processed\masks\livecell_train\val\val\SH-SHY5Y\SHSY5Y_Phase_C10_2_01d04h00m_4.tif"

# # Load the binary mask
# binary_mask = tiff.imread(file_path)

# # Save the binary mask matrix to a text file
# output_text_file = "binary_mask_matrix.txt"
# with open(output_text_file, "w") as f:
#     for row in binary_mask:
#         np.savetxt(f, row[None], fmt="%d")  # Write each row as integers
# print(f"Full binary mask matrix saved to '{output_text_file}'")

# # Save the binary mask as an image
# output_image_file = "binary_mask_image.png"
# plt.imsave(output_image_file, binary_mask, cmap="gray")
# print(f"Binary mask saved as an image to '{output_image_file}'")

# # Optional: Display the saved image for confirmation
# plt.imshow(binary_mask, cmap="gray")
# plt.title("Binary Mask Image")
# plt.axis("off")
# plt.show()


# import tifffile as tiff
# import numpy as np
# import sys

# # Function to check for required dependencies
# def check_imagecodecs():
#     try:
#         import imagecodecs  # noqa
#         return True
#     except ImportError:
#         print("The 'imagecodecs' package is not installed. Install it using 'pip install imagecodecs'.")
#         return False

# # File paths
# tif_image_path = r"E:\diploma_proj_latest\cell-segmentation\.data\03_processed\masks\livecell_test\test\A172\A172_Phase_C7_1_00d00h00m_1.tif"
# binary_mask_txt_path = r"E:\diploma_proj_latest\cell-segmentation\binary_mask_matrix.txt"

# # Run only if 'imagecodecs' is installed
# if check_imagecodecs():
#     try:
#         # Load the TIFF image
#         tif_image = tiff.imread(tif_image_path)

#         # Load the binary mask matrix from the text file
#         binary_mask_matrix = np.loadtxt(binary_mask_txt_path, dtype=int)

#         # Check if dimensions match
#         tif_shape = tif_image.shape
#         mask_shape = binary_mask_matrix.shape

#         # Output results
#         print("TIFF Image Shape:", tif_shape)
#         print("Binary Mask Matrix Shape:", mask_shape)
#         print("Shapes Match:", tif_shape == mask_shape)

#         # Verify pixel correspondence
#         if tif_shape == mask_shape:
#             print("Dimensions match. Checking pixel correspondence...")
#             pixel_difference = np.sum(np.abs(tif_image - binary_mask_matrix))
#             print("Total Pixel Difference:", pixel_difference)
#             print("Mask corresponds to image." if pixel_difference == 0 else "Mask does NOT correspond to image.")
#         else:
#             print("Dimensions do NOT match. Cannot verify correspondence.")

#     except Exception as e:
#         print(f"Error: {e}")
#         print("Ensure the TIFF image is uncompressed or install the 'imagecodecs' package.")
# else:
#     print("Cannot proceed without 'imagecodecs'.")


import numpy as np
import tifffile as tiff

image_path = r"E:\diploma_proj_latest\cell-segmentation\.data\03_processed\masks\livecell_test\test\A172\A172_Phase_C7_1_00d00h00m_1.tif"
normalizedd=  r"E:\diploma_proj_latest\cell-segmentation\.data\03_processed\normalized\livecell_train\train\A172\A172_Phase_A7_1_00d00h00m_1.npy"
# Load an example
image = np.load(normalizedd)
mask = tiff.imread(image_path)

print("Image shape:", image.shape)  # Should be (H, W) or (1, H, W)
print("Mask unique values:", np.unique(mask))  # Should print [0 1 2 3 4 5 6 7]