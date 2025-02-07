from kedro.pipeline import Pipeline, node
from .nodes import (
    resize_images,
    normalize_images_for_train,
    normalize_images_for_test,
    create_masks_for_test,
    create_masks_for_train,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            # Resize train images
            node(
                func=resize_images,
                inputs=["params:input_folder_train", "params:output_folder_train", "params:target_size"],
                outputs="resized_train_folder",
                name="resize_train_images",
            ),
            #  Resize test images
            node(
                func=resize_images,
                inputs=["params:input_folder_test", "params:output_folder_test", "params:target_size"],
                outputs="resized_test_folder",
                name="resize_test_images",
            ),
            #  Normalize train images
            node(
                func=normalize_images_for_train,
                inputs=["resized_train_folder", "params:norm_output_folder_train", "params:split_parameters", "params:normalization"],
                outputs="normalized_train_data",
                name="normalize_images_train",
            ),
            #  Normalize test images
            node(
                func=normalize_images_for_test,
                inputs=["resized_test_folder", "params:norm_output_folder_test", "params:normalization"],
                outputs="normalized_test_data",
                name="normalize_images_test",
            ),
            #  Create train masks
            node(
                func=create_masks_for_train,
                inputs=["resized_train_folder", "params:mask_output_folder_train", "params:split_parameters", "params:thresholding_parameters"],
                outputs="train_masks_data",
                name="create_masks_train",
            ),
            #  Create test masks
            node(
                func=create_masks_for_test,
                inputs=["resized_test_folder", "params:mask_output_folder_test","params:thresholding_parameters"],
                outputs="test_masks_data",
                name="create_masks_test",
            ),
        ]
    )
