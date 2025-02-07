# import numpy as np
# import tifffile as tiff
# from pathlib import Path
# from torch.utils.data import DataLoader
# from .tools import SegmentationDataset

# # from .u_net import UNet


# def verify_dataset_integrity() :
#     # Load a few sample masks
#     mask_files = list(Path("E:/diploma_proj_latest/cell-segmentation/.data/02_masks/train").glob("*.tif"))

#     for mask_path in mask_files[:5]:  # Check first 5 masks
#         mask = tiff.imread(mask_path)
#         unique_values = np.unique(mask)
#         print(f"Mask {mask_path.name} unique values: {unique_values}")

#         # Ensure all values are in [0, 1, ..., 7]
#         assert set(unique_values).issubset(set(range(8))), f"❌ Mask {mask_path} contains unexpected values: {unique_values}"

#     print("✅ All masks are correctly labeled in range [0..7].")


# def check_data_pipeline() :
#     # Load dataset
#     train_images_dir = Path("E:/diploma_proj_latest/cell-segmentation/.data/01_images/train")
#     train_masks_dir = Path("E:/diploma_proj_latest/cell-segmentation/.data/02_masks/train")

#     dataset = SegmentationDataset(train_images_dir, train_masks_dir)
#     dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

#     # Fetch one batch
#     images, masks = next(iter(dataloader))

#     # Assert image and mask shapes
#     assert images.shape == (4, 1, 512, 512), f"❌ Expected image shape (4, 1, 512, 512), got {images.shape}"
#     assert masks.shape == (4, 512, 512), f"❌ Expected mask shape (4, 512, 512), got {masks.shape}"

#     print("✅ DataLoader returns correct shapes.")

# # def check_model_output_shape() :
# #     model = UNet(num_classes=8)
# #     outputs = model(images)  # Forward pass

# #     assert outputs.shape == (4, 8, 512, 512), f"❌ Expected model output shape (4, 8, 512, 512), got {outputs.shape}"

# #     print("✅ Model outputs correct shape.")    


# import numpy as np
# import tifffile as tiff

# sample_mask_path_1 = r"E:\diploma_proj_latest\cell-segmentation\.data\03_processed\masks\livecell_test\test\BT474\BT474_Phase_D3_1_00d00h00m_1.tif"
# sample_mask_path_2 =  r'E:\diploma_proj_latest\cell-segmentation\.data\03_processed\masks\livecell_test\test\BV-2\BV2_Phase_A4_1_00d00h00m_1.tif'


# mask = tiff.imread(sample_mask_path_2)
# unique_values = np.unique(mask)
# print(f"Unique values in mask: {unique_values}")

# Attempt to read and analyze the newly uploaded image
import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt

# Load the provided image
image_path = r"E:\diploma_proj_latest\cell-segmentation\.data\02_intermediate\livecell_test\A172\A172_Phase_C7_1_00d00h00m_1.tif"
image = tiff.imread(image_path)

# Analyze the unique intensity values
unique_values = np.unique(image)
min_intensity, max_intensity = image.min(), image.max()

# Display unique values and intensity range
analysis_result = {
    "Unique Values": unique_values,
    "Min Intensity": min_intensity,
    "Max Intensity": max_intensity
}

# Plot the image
plt.figure(figsize=(6,6))
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.colorbar()
plt.show()

# Plot histogram of intensity values
plt.figure(figsize=(6,4))
plt.hist(image.flatten(), bins=50, color="blue", alpha=0.7)
plt.title("Histogram of Pixel Intensities")
plt.xlabel("Intensity")
plt.ylabel("Frequency")
plt.show()