"""Main module for the project."""

import json

from comparator.config import CONFIG
from comparator.dataset import DatasetManager
from comparator.trainer import RailTrackModelComparison


def is_colab() -> bool:
    """Check if running in Google Colab."""
    try:
        from google import colab

        return True
    except ImportError:
        return False


def main():
    # Initialize with Google Drive if in Colab
    use_drive = is_colab()

    # Initialize DatasetManager
    dataset_manager = DatasetManager(CONFIG)

    # Download datasets in both formats
    yolo_path = dataset_manager.download(data_format="yolov8")
    coco_path = dataset_manager.download(data_format="coco")

    # Initialize model comparison
    trainer = RailTrackModelComparison(CONFIG, use_drive=use_drive)

    try:
        # Check for previous checkpoint
        checkpoint_file = trainer.checkpoint_file
        if checkpoint_file.exists():
            with open(checkpoint_file, "r") as f:
                checkpoint = json.load(f)
                last_completed = checkpoint.get("last_completed", "yolov8")
                print(f"Resuming from checkpoint. Last completed: {last_completed}")
                print(f"Checkpoint timestamp: {checkpoint.get('timestamp', 'unknown')}")
        else:
            last_completed = None

        # Train models, respecting checkpoint
        if last_completed != "yolov8":
            trainer.train_yolov8(yolo_path)
        if last_completed != "detectron2":
            trainer.train_detectron2(coco_path)

        # Compare models
        comparison_results = trainer.compare_models()

        # Print results
        print("\nModel Comparison Results:")
        print("-" * 50)
        print("\nTraining Time Comparison:")
        for model, time in comparison_results["training_time_comparison"].items():
            print(f"{model}: {time:.2f} seconds")

        print("\nMemory Usage Comparison:")
        for model, memory in comparison_results["memory_usage_comparison"].items():
            print(f"{model}: {memory / 1e9:.2f} GB")

        print("\nModel Paths:")
        for model, path in comparison_results["model_paths"].items():
            print(f"{model}: {path}")

        if "yolov8_map" in comparison_results:
            print(f"\nYOLOv8 mAP: {comparison_results['yolov8_map']:.4f}")

    except KeyboardInterrupt:
        print("\nTraining interrupted. Progress has been saved.")
        print("You can resume training by running the script again.")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        print("Progress has been saved and can be resumed.")
    finally:
        if use_drive:
            print(
                "\nAll results have been saved to Google Drive under 'rail_track_comparison' folder."
            )
            print("You can access them even if the Colab session ends.")


if __name__ == "__main__":
    main()
