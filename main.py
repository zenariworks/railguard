"""Main module for the project."""

from config import CONFIG
from dataset import DatasetManager
from models.model_trainers import ModelTrainer
from utils import Evaluator


def main():
    # Initialize dataset manager
    dataset_manager = DatasetManager(CONFIG)

    # Initialize model trainer
    trainer = ModelTrainer(CONFIG, dataset_manager)

    # Initialize evaluator
    evaluator = Evaluator(CONFIG)

    # Train and evaluate models
    models_to_train = {
        "YOLOv8": trainer.train_yolov8,
        "Detectron2": trainer.train_detectron2,
        "BiSeNet": trainer.train_bisenet,
        "DeepLabV3": trainer.train_deeplabv3,
    }

    for model_name, train_func in models_to_train.items():
        try:
            print(f"\nTraining {model_name}...")
            model_data = train_func()
            evaluator.evaluate_model(model_name, model_data, dataset_manager)
        except Exception as e:
            print(f"Error training {model_name}: {str(e)}")

    # Generate final report
    report_path = evaluator.generate_report()
    print(f"\nComparison report saved to: {report_path}")
    print("Visualization saved to: results/comparison_plots.png")


if __name__ == "__main__":
    main()
