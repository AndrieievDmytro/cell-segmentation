from kedro.pipeline import Pipeline, node
from .nodes import (
    resize_images,
    normalize_images_for_train,
    normalize_images_for_test,
    create_masks_for_test,
    create_masks_for_train,
    # create_masks,
    # normalize_images,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            # Step 1: Resize train images
            node(
                func=resize_images,
                inputs=["params:input_folder_train", "params:output_folder_train", "params:target_size"],
                outputs="resized_train_folder",
                name="resize_train_images",
            ),
            # Step 2: Resize test images
            node(
                func=resize_images,
                inputs=["params:input_folder_test", "params:output_folder_test", "params:target_size"],
                outputs="resized_test_folder",
                name="resize_test_images",
            ),
            # Step 3: Normalize train images
            node(
                func=normalize_images_for_train,
                inputs=["resized_train_folder", "params:norm_output_folder_train", "params:split_parameters"],
                outputs="normalized_train_data",
                name="normalize_images_train",
            ),
            # Step 4: Normalize test images
            node(
                func=normalize_images_for_test,
                inputs=["resized_test_folder", "params:norm_output_folder_test"],
                outputs="normalized_test_data",
                name="normalize_images_test",
            ),
            # Step 5: Create train masks
            node(
                func=create_masks_for_train,
                inputs=["resized_train_folder", "params:mask_output_folder_train", "params:split_parameters"],
                outputs="train_masks_data",
                name="create_masks_train",
            ),
            # Step 6: Create test masks
            node(
                func=create_masks_for_test,
                inputs=["resized_test_folder", "params:mask_output_folder_test"],
                outputs="test_masks_data",
                name="create_masks_test",
            ),
        ]
    )
