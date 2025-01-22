"""Project pipelines."""

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from .pipelines.data import pipeline as data_pipeline # Import the data_processing pipeline

# from cell_analysis.pipelines.data import pipeline as data_pipeline  

def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    # Dynamically discover pipelines
    pipelines = find_pipelines()

    # Add the data_processing pipeline explicitly
    pipelines["data_processing"] = data_pipeline.create_pipeline()

    # Set the default pipeline to sum of all pipelines
    pipelines["__default__"] = sum(pipelines.values())

    return pipelines
