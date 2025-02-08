from kedro.pipeline import Pipeline, node
from .nodes import train_model
# from 


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        # Train the model
    node(
        func=train_model,
        inputs=["normalized_train_data", "train_masks_data", "params:training_parameters"],
        outputs="trained_model",
        name="train_model",
    )      
    ])
