# from PIL import Image
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

from pathlib import Path

def check_image_mask_correspondence(normalized_train_folder, masks_folder):
    """
    Check that all images in the normalized train folder have corresponding masks
    and vice versa.
    """
    normalized_train_path = Path(normalized_train_folder)
    masks_path = Path(masks_folder)
    
    mismatched_files = []

    # Check each class folder
    for class_folder in normalized_train_path.iterdir():
        if class_folder.is_dir():
            class_name = class_folder.name
            class_masks_folder = masks_path / class_name
            
            # Check if the class folder exists in the masks folder
            if not class_masks_folder.exists():
                print(f"Class folder {class_name} is missing in the masks directory.")
                mismatched_files.append((class_name, "Missing mask class folder"))
                continue
            
            # List files in both folders
            train_files = {file.stem for file in class_folder.glob("*.npy")}
            mask_files = {file.stem for file in class_masks_folder.glob("*.npy")}
            
            # Find mismatches
            missing_masks = train_files - mask_files
            missing_images = mask_files - train_files
            
            # Report mismatches
            if missing_masks:
                print(f"Missing masks for class '{class_name}': {missing_masks}")
                mismatched_files.extend([(class_name, f"Missing mask: {name}.npy") for name in missing_masks])
            
            if missing_images:
                print(f"Missing images for class '{class_name}': {missing_images}")
                mismatched_files.extend([(class_name, f"Missing image: {name}.npy") for name in missing_images])
    
    if not mismatched_files:
        print("All image-mask correspondences are correct.")
    else:
        print("Some mismatches were found.")
    
    return mismatched_files

# Example usage
normalized_train_folder = "E:/diploma_proj_latest/cell-segmentation/.data/03_processed/normalized/livecell_train"
masks_folder = "E:/diploma_proj_latest/cell-segmentation/.data/03_processed/masks/livecell_train"

mismatches = check_image_mask_correspondence(normalized_train_folder, masks_folder)
