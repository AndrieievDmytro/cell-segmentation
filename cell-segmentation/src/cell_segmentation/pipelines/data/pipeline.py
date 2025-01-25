from kedro.pipeline import Pipeline, node
from .nodes import resize_images, create_masks, normalize_images

def create_pipeline(**kwargs):
    return Pipeline([
        # Step 1: Resize train images
        node(
            func=resize_images,
            inputs=dict(
                input_folder="params:input_folder_train",
                output_folder="params:output_folder_train",
                target_size="params:target_size"
            ),
            outputs="resized_train_folder",
            name="resize_train_images",
        ),

        # Step 2: Create masks for train images
        node(
            func=create_masks,
            inputs=dict(
                input_folder="resized_train_folder",  # Depends on the output of Step 1
                output_folder="params:mask_output_folder_train"
            ),
            outputs="train_masks_folder",
            name="create_train_masks",
        ),

        # Step 3: Normalize train images
        node(
            func=normalize_images,
            inputs=dict(
                input_folder="train_masks_folder",  # Depends on the output of Step 2
                output_folder="params:norm_output_folder_train",
                mean="params:normalization.mean",
                std="params:normalization.std"
            ),
            outputs=None,  # Final output, no further dependency
            name="normalize_train_images",
        ),

        # Step 4: Resize test images
        node(
            func=resize_images,
            inputs=dict(
                input_folder="params:input_folder_test",
                output_folder="params:output_folder_test",
                target_size="params:target_size"
            ),
            outputs="resized_test_folder",
            name="resize_test_images",
        ),

        # Step 5: Create masks for test images
        node(
            func=create_masks,
            inputs=dict(
                input_folder="resized_test_folder",  # Depends on the output of Step 4
                output_folder="params:mask_output_folder_test"
            ),
            outputs="test_masks_folder",
            name="create_test_masks",
        ),

        # Step 6: Normalize test images
        node(
            func=normalize_images,
            inputs=dict(
                input_folder="test_masks_folder",  # Depends on the output of Step 5
                output_folder="params:norm_output_folder_test",
                mean="params:normalization.mean",
                std="params:normalization.std"
            ),
            outputs=None,  # Final output, no further dependency
            name="normalize_test_images",
        ),
    ])
