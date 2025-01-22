# from kedro.pipeline import Pipeline, node
# from .nodes import resize_images

# def create_pipeline(**kwargs) -> Pipeline:
#     return Pipeline([
#         node(
#             func=resize_images,
#             inputs=["livecell_test_raw", "livecell_test_resized", "params:target_size"],
#             outputs=None,
#             name="resize_test_images"
#         ),
#         node(
#             func=resize_images,
#             inputs=["livecell_train_raw", "livecell_train_resized", "params:target_size"],
#             outputs=None,
#             name="resize_train_images"
#         )
#     ])

from kedro.pipeline import Pipeline, node
from .nodes import resize_images

def create_pipeline(**kwargs):
    return Pipeline([
        node(
            func=resize_images,
            inputs=dict(
                input_folder="params:input_folder_test",
                output_folder="params:output_folder_test",
                target_size="params:target_size"
            ),
            outputs=None,
            name="resize_test_images",
        ),
        node(
            func=resize_images,
            inputs=dict(
                input_folder="params:input_folder_train",
                output_folder="params:output_folder_train",
                target_size="params:target_size"
            ),
            outputs=None,
            name="resize_train_images",
        ),
    ])
