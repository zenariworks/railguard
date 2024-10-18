"""Main module for the project."""
from config import CONFIG
from dataset import DatasetManager
from models.model_trainer import RailTrackModelComparison


def main():
    # Initialize DatasetManager
    dataset_manager = DatasetManager(CONFIG)

    # Download YOLOv8 format
    yolo_path = dataset_manager.download_dataset(data_format="yolov8")

    # Download COCO format
    coco_path = dataset_manager.download_dataset(data_format="coco")

    # Initialize comparison framework
    comparison = RailTrackModelComparison(CONFIG)

    # Run complete comparison
    comparison.run_comparison()


if __name__ == "__main__":
    main()
