from pathlib import Path
from PIL import Image
import numpy as np

def resize_images(input_folder, output_folder, target_size):
    width, height = int(target_size["width"]), int(target_size["height"])
    target_size = (width, height)

    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    for file in input_path.glob("*.tif"):
        try:
            with Image.open(file) as img:
                resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
                output_file = output_path / file.name
                resized_img.save(output_file, format="TIFF", compression="tiff_lzw")
                print(f"Resized and saved: {file.name}")
        except Exception as e:
            print(f"Error processing {file.name}: {e}")

    return str(output_path)  # Return the output folder path


def create_masks(input_folder, output_folder, threshold=128):
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    for file in input_path.glob("*.tif"):
        try:
            with Image.open(file) as img:
                gray = img.convert("L")
                arr = np.array(gray, dtype=np.uint8)
                mask_arr = np.where(arr > threshold, 255, 0).astype(np.uint8)
                mask_img = Image.fromarray(mask_arr, mode="L")
                output_file = output_path / file.name
                mask_img.save(output_file, format="TIFF", compression="tiff_lzw")
                print(f"Mask created: {file.name}")
        except Exception as e:
            print(f"Error processing {file.name}: {e}")

    return str(output_path)  # Return the output folder path


def normalize_images(input_folder, output_folder, mean=0.5, std=0.2):
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    for file in input_path.glob("*.tif"):
        try:
            with Image.open(file) as img:
                arr = np.array(img, dtype=np.float32)
                arr = arr / 255.0
                arr = (arr - mean) / std
                arr = np.clip(arr, -3.0, 3.0)
                arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-7) * 255.0
                arr = arr.astype(np.uint8)
                norm_img = Image.fromarray(arr)
                output_file = output_path / file.name
                norm_img.save(output_file, format="TIFF", compression="tiff_lzw")
                print(f"Normalized and saved: {file.name}")
        except Exception as e:
            print(f"Error processing {file.name}: {e}")

    return str(output_path)  # Return the output folder path

