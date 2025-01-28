from kedro.pipeline import Pipeline, node
from .nodes import train_model


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=train_model,
            inputs=["train_data", "val_data", "params:training_parameters"],
            outputs="trained_model",
            name="train_model",
        ),
    ])
