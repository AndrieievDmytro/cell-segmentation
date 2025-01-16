import cv2
import numpy as np
import os

def load_images_and_masks(data_dir, img_size=(128, 128)):
    images, masks = [], []

    for subset in ["train", "test"]:
        img_dir = os.path.join(data_dir, subset, "images")
        mask_dir = os.path.join(data_dir, subset, "masks")

        for img_name in os.listdir(img_dir):
            img_path = os.path.join(img_dir, img_name)
            mask_path = os.path.join(mask_dir, img_name)

            # Read and resize images and masks
            image = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            image = cv2.resize(image, img_size)
            mask = cv2.resize(mask, img_size)

            images.append(image)
            masks.append(mask)

    images = np.array(images, dtype=np.float32) / 255.0
    masks = np.array(masks, dtype=np.float32) / 255.0

    return images, masks
