from pathlib import Path
from kedro.pipeline import Pipeline
from .pipelines.data import pipeline as data_processing  
from .pipelines.model import pipeline as model_training  
from .pipelines.evaluation import pipeline as evaluation_pipeline

def register_pipelines() -> dict[str, Pipeline]:
    # Explicitly define pipelines
    data_pipeline = data_processing.create_pipeline()
    model_pipeline = model_training.create_pipeline()
    eval_pipeline = evaluation_pipeline.create_pipeline()

    full_pipeline = data_pipeline + model_pipeline + eval_pipeline

    return {
        "data_processing": data_pipeline,
        "model_training": model_pipeline,
        "full_pipeline": full_pipeline,
        "eval_pipeline": eval_pipeline,
        "__default__": full_pipeline,
    }

    