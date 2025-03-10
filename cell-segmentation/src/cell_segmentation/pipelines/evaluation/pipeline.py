from kedro.pipeline import Pipeline, node
from .nodes import evaluate_model


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        # Evaluation node
        node(
            func=evaluate_model,
            inputs=["params:evaluation_parameters"],
            outputs="evaluation_metrics",
            name="evaluate_model",
        ),
    ])
