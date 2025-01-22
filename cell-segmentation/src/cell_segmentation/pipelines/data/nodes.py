# from PIL import Image
# import os

# def resize_images(input_folder, output_folder, target_size):
#     """
#     Resizes all .tif images in the input folder and saves them to the output folder with minimal quality loss.

#     Args:
#         input_folder (str): Path to the folder containing input .tif images.
#         output_folder (str): Path to the folder to save resized .tif images.
#         target_size (dict): Dictionary containing desired size with keys 'width' and 'height'.
#     """
#     target_size = (target_size["width"], target_size["height"])  # Convert to tuple
#     os.makedirs(output_folder, exist_ok=True)  # Create output folder if it doesn't exist

#     for filename in os.listdir(input_folder):
#         if filename.endswith(".tif"):
#             input_path = os.path.join(input_folder, filename)
#             output_path = os.path.join(output_folder, filename)

#             # Open the image
#             with Image.open(input_path) as img:
#                 # Resize with high-quality resampling
#                 resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
                
#                 # Save with lossless compression
#                 resized_img.save(output_path, format="TIFF", compression="tiff_lzw", save_all=True)

#                 print(f"Resized {filename} to {target_size} and saved as {output_path}")

from pathlib import Path
from PIL import Image

# def resize_images(input_folder, output_folder, target_size):
#     """
#     Resize all .tif images in the input folder and save them to the output folder.
#     """
#     input_path = Path(input_folder)
#     output_path = Path(output_folder)
#     output_path.mkdir(parents=True, exist_ok=True)  # Ensure output folder exists

#     for file in input_path.glob("*.tif"):
#         try:
#             with Image.open(file) as img:
#                 # Resize the image
#                 resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
#                 # Save the image
#                 resized_img.save(output_path / file.name, format="TIFF", compression="tiff_lzw")
#                 print(f"Resized and saved: {file.name}")
#         except Exception as e:
#             print(f"Error processing {file.name}: {e}")

def resize_images(input_folder, output_folder, target_size):
    # Ensure target_size values are integers
    width, height = int(target_size["width"]), int(target_size["height"])
    target_size = (width, height)

    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    for file in input_path.glob("*.tif"):
        try:
            with Image.open(file) as img:
                resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
                resized_img.save(output_path / file.name, format="TIFF", compression="tiff_lzw")
                print(f"Resized and saved: {file.name}")
        except Exception as e:
            print(f"Error processing {file.name}: {e}")
