import cv2
import tifffile as tiff
import numpy as np
from pathlib import Path
import random
from typing import List, Tuple
from skimage.morphology import remove_small_objects


def split_data(files: List[Path], train_ratio: float, random_seed: int) -> Tuple[List[Path], List[Path]]:

    random.seed(random_seed)
    random.shuffle(files)
    split_idx = int(len(files) * train_ratio)
    train_files = files[:split_idx]
    val_files = files[split_idx:]
    return train_files, val_files

def enhance_contrast(image):

    if len(image.shape) > 2:  
        raise ValueError("CLAHE expects a single-channel grayscale image, but got a multi-channel image.")
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image.astype(np.uint8))  # Ensure uint8 dtype for CLAHE
    return enhanced


def threshold_and_morphology(enhanced_image, threshold_val=40, use_adaptive=False, use_otsu=False,
                            invert_threshold=True, min_obj_size=50):
    # Choose the thresholding method:
    if use_adaptive:
        if invert_threshold:
            binary_mask = cv2.adaptiveThreshold(enhanced_image, 255,
                                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                    cv2.THRESH_BINARY_INV, 11, 2)
        else:
            binary_mask = cv2.adaptiveThreshold(enhanced_image, 255,
                                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                    cv2.THRESH_BINARY, 11, 2)
    elif use_otsu:
        if invert_threshold:
            ret, binary_mask = cv2.threshold(enhanced_image, 0, 255,
                                                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            ret, binary_mask = cv2.threshold(enhanced_image, 0, 255,
                                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        if invert_threshold:
            ret, binary_mask = cv2.threshold(enhanced_image, threshold_val, 255, cv2.THRESH_BINARY_INV)
        else:
            ret, binary_mask = cv2.threshold(enhanced_image, threshold_val, 255, cv2.THRESH_BINARY)
    
    # Convert to boolean mask:
    binary_mask = binary_mask > 0


    # # Optionally remove small objects (minimal cleanup)
    binary_mask_clean = remove_small_objects(binary_mask, min_size=min_obj_size)
    
    return binary_mask_clean


def process_class_folder(class_folder, output_path, thresholding_parameters, mode, specific_files=None):

    # --- Set thresholding parameters ---
    use_adaptive = thresholding_parameters["use_adaptive"]    
    use_otsu = thresholding_parameters["use_otsu"]        
    invert_threshold = thresholding_parameters["invert_threshold"] 
    threshold_val = thresholding_parameters["threshold_val"]      
    min_obj_size = thresholding_parameters["min_obj_size"]       

    folder_to_class_id = {
        "A172": 1,
        "BT474": 2,
        "BV-2": 3,
        "Huh7": 4,
        "MCF7": 5,
        "SH-SHY5Y": 6,
        "SK-OV-3": 7,
        "SkBr3": 8,
    }
    class_id = folder_to_class_id.get(class_folder.name, -1)  # Default to -1 if folder is not mapped
    if class_id == -1:
        print(f"Warning: Folder {class_folder.name} is not mapped to a class ID. Skipping...")
        return

    output_folder = output_path / mode / class_folder.name
    output_folder.mkdir(parents=True, exist_ok=True)

    files = specific_files if specific_files else class_folder.glob("*.tif")
    for mask_path in files:
        mask = tiff.imread(mask_path)
        enhanced_image = enhance_contrast(mask)
        
        # Apply thresholding and minimal cleanup:
        binary_mask = threshold_and_morphology(enhanced_image, threshold_val, use_adaptive,
                                        use_otsu, invert_threshold, min_obj_size)
        # Save the processed mask
        output_mask_path = output_folder / mask_path.name
        binary_mask = binary_mask.astype(np.uint16) * class_id  
        tiff.imwrite(output_mask_path, binary_mask)


