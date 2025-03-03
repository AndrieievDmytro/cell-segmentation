from pathlib import Path
from PIL import Image
import numpy as np
import random
from .tools import  split_data, process_class_folder

def resize_images(input_folder, output_folder, target_size):

    width, height = int(target_size["width"]), int(target_size["height"])
    target_size = (width, height)

    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # Iterate through class folders in the input path
    for class_folder in input_path.iterdir():
        if class_folder.is_dir():
            class_name = class_folder.name
            output_class_folder = output_path / class_name
            output_class_folder.mkdir(parents=True, exist_ok=True)

            # Process all .tif files in the class folder
            for file in class_folder.glob("*.tif"):
                try:
                    with Image.open(file) as img:
                        resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
                        output_file = output_class_folder / file.name
                        resized_img.save(output_file, format="TIFF", compression="tiff_lzw")
                        # print(f"Resized and saved: {output_file}")
                except Exception as e:
                    print(f"Error processing {file.name}: {e}")

    return str(output_path)  # Return the output folder path


def normalize_image_array(arr, mean=0.5, std=0.2):

    # Normalize to [0, 1]
    arr = arr / 255.0

    # Apply mean and std normalization
    arr = (arr - mean) / std

    # Clip to a reasonable range
    arr = np.clip(arr, -3.0, 3.0)

    return arr

def save_normalized_array(arr, output_path):

    np.save(output_path, arr)  # Save as .npy file

def normalize_image_array(arr, mean, std):

    arr = arr / 255.0  # Scale to [0, 1]
    arr = (arr - arr.mean()) / (arr.std() + 1e-7)  # Normalize to zero mean, unit variance
    arr = arr * std + mean  # Scale to desired mean and std
    arr = np.clip(arr, 0, 1)  # Clip values to [0, 1]
    return arr

def normalize_images_for_test(input_folder, output_folder, normalization):

    print(f"Input folder: {input_folder}, Output folder: {output_folder}")

    mean = normalization["mean"]
    std = normalization["std"]

    print(f"mean: {mean}, std: {std}")

    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    for class_folder in input_path.iterdir():
        if class_folder.is_dir():
            class_name = class_folder.name
            output_class_folder = output_path / class_name
            output_class_folder.mkdir(parents=True, exist_ok=True)

            print(f"Processing class: {class_name}")
            for file in class_folder.glob("*.tif"):
                try:
                    with Image.open(file) as img:
                        arr = np.array(img, dtype=np.float32)
                        arr = normalize_image_array(arr, mean, std)
                        save_normalized_array(arr, output_class_folder / (file.stem + ".npy"))
                        # print(f"Test normalized image saved: {file.stem}.npy")
                except Exception as e:
                    print(f"Error processing {file.name}: {e}")

    print("Normalization for test data completed.")
    return str(output_path)

def normalize_images_for_train(input_folder, output_folder, split_parameters, normalization):

    print(f"Input folder: {input_folder}, Output folder: {output_folder}")

    mean = normalization["mean"]
    std = normalization["std"]

    print(f"mean: {mean}, std: {std}")

    input_path = Path(input_folder)
    output_path = Path(output_folder)

    train_ratio = split_parameters["train_ratio"]
    random_seed = split_parameters["random_seed"]
    random.seed(random_seed)

    train_output_path = output_path / "train"
    val_output_path = output_path / "val"
    train_output_path.mkdir(parents=True, exist_ok=True)
    val_output_path.mkdir(parents=True, exist_ok=True)

    for class_folder in input_path.iterdir():
        if class_folder.is_dir():
            class_name = class_folder.name
            train_class_folder = train_output_path / class_name
            val_class_folder = val_output_path / class_name
            train_class_folder.mkdir(parents=True, exist_ok=True)
            val_class_folder.mkdir(parents=True, exist_ok=True)

            print(f"Processing class: {class_name}")
            files = sorted(list(class_folder.glob("*.tif")))

            # Split data into train and validation sets
            split_idx = int(len(files) * train_ratio)
            random.shuffle(files)
            train_files = files[:split_idx]
            val_files = files[split_idx:]

            # Normalize and save train images
            for file in train_files:
                try:
                    with Image.open(file) as img:
                        arr = np.array(img, dtype=np.float32)
                        arr = normalize_image_array(arr, mean, std)
                        save_normalized_array(arr, train_class_folder / (file.stem + ".npy"))
                        # print(f"Train normalized image saved: {file.stem}.npy")
                except Exception as e:
                    print(f"Error processing {file.name}: {e}")

            # Normalize and save validation images
            for file in val_files:
                try:
                    with Image.open(file) as img:
                        arr = np.array(img, dtype=np.float32)
                        arr = normalize_image_array(arr, mean, std)
                        save_normalized_array(arr, val_class_folder / (file.stem + ".npy"))
                        # print(f"Validation normalized image saved: {file.stem}.npy")
                except Exception as e:
                    print(f"Error processing {file.name}: {e}")

    print("Normalization and splitting for training data completed.")
    return str(output_path)

def create_masks_for_test(input_folder, output_folder, thresholding_parameters):

    print(
    "\n--- Thresholding Parameters ---\n"
    f"Use Adaptive Thresholding: {thresholding_parameters['use_adaptive']}\n"
    f"Use Otsu's Method: {thresholding_parameters['use_otsu']}\n"
    f"Invert Thresholding: {thresholding_parameters['invert_threshold']}\n"
    f"Threshold Value (if not using Otsu): {thresholding_parameters['threshold_val']}\n"
    f"Minimum Object Size: {thresholding_parameters['min_obj_size']}\n"
    "-------------------------------"
    )


    print(f"Input folder: {input_folder}, Output folder: {output_folder}")
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    for class_folder in input_path.iterdir():
        if class_folder.is_dir():
            process_class_folder(class_folder, output_path, thresholding_parameters, mode="test")

    print("Mask creation for test data completed.")
    return str(output_path)



def create_masks_for_train(input_folder, output_folder, split_parameters, thresholding_parameters):

    print(f"Input folder: {input_folder}, Output folder: {output_folder}")
    input_path = Path(input_folder)
    output_path = Path(output_folder)

    train_ratio = split_parameters["train_ratio"]
    random_seed = split_parameters["random_seed"]

    train_output_path = output_path 
    val_output_path = output_path 
    train_output_path.mkdir(parents=True, exist_ok=True)
    val_output_path.mkdir(parents=True, exist_ok=True)

    for class_folder in input_path.iterdir():
        if class_folder.is_dir():
            files = sorted(list(class_folder.glob("*.tif")))
            train_files, val_files = split_data(files, train_ratio, random_seed)

            process_class_folder(class_folder, train_output_path, thresholding_parameters, mode="train", specific_files=train_files)
            process_class_folder(class_folder, val_output_path, thresholding_parameters, mode="val", specific_files=val_files)

    print("Mask creation and splitting for training data completed.")
    return str(output_path)

