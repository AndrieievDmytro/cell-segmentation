"""Project pipelines."""

from pathlib import Path
from kedro.pipeline import Pipeline
from .pipelines.data import pipeline as data_processing  # Import data pipeline
from .pipelines.model import pipeline as model_training  # Import model training pipeline

def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    # Returns:
    #     A mapping from pipeline names to ``Pipeline`` objects.
    # """
    # # Explicitly define pipelines
    # data_pipeline = data_processing.create_pipeline()
    # model_pipeline = model_training.create_pipeline()

    # # Connect pipelines
    # full_pipeline = data_pipeline + model_pipeline

    # return {
    #     "data_processing": data_pipeline,
    #     "model_training": model_pipeline,
    #     "full_pipeline": full_pipeline,
    #     "__default__": full_pipeline,
    # }
    # Paths to preprocessed files
    # train_images_dir = Path("E:/diploma_proj_latest/cell-segmentation/.data/03_processed/normalized/livecell_train/train")
    # train_masks_dir = Path("E:/diploma_proj_latest/cell-segmentation/.data/03_processed/masks/livecell_train/train")
    # val_images_dir = Path("E:/diploma_proj_latest/cell-segmentation/.data/03_processed/normalized/livecell_train/val")
    # val_masks_dir = Path("E:/diploma_proj_latest/cell-segmentation/.data/03_processed/masks/livecell_train/val")

    # # Check if preprocessed files exist
    # files_exist = all([
    #     train_images_dir.exists(),
    #     train_masks_dir.exists(),
    #     val_images_dir.exists(),
    #     val_masks_dir.exists()
    # ])

    # Explicitly define pipelines
    data_pipeline = data_processing.create_pipeline()
    model_pipeline = model_training.create_pipeline()

    # # Determine the full pipeline
    # if files_exist:
    #     print("Preprocessed files found. Skipping data_processing pipeline.")
    #     full_pipeline = model_pipeline
    # else:
    #     print("Preprocessed files not found. Running full pipeline (data_processing + model_training).")
    full_pipeline = data_pipeline + model_pipeline

    return {
        "data_processing": data_pipeline,
        "model_training": model_pipeline,
        "full_pipeline": full_pipeline,
        "__default__": full_pipeline,
    }

    