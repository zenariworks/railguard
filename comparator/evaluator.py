import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import seaborn as sns
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class EnhancedEvaluator:
    def __init__(self, config: dict, output_dir: Path):
        """
        Initialize the evaluator with configuration and output directory.

        Args:
            config: Configuration dictionary
            output_dir: Path to output directory
        """
        self.config = config
        self.output_dir = output_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = {}
        self.writer = SummaryWriter(output_dir / "tensorboard")

        # Create necessary directories
        self.models_dir = output_dir / "saved_models"
        self.results_dir = output_dir / "results"
        self.plots_dir = output_dir / "plots"

        for dir_path in [self.models_dir, self.results_dir, self.plots_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def get_resource_usage(self) -> dict[str, float]:
        """Get current system resource usage."""
        gpu_memory_used = 0
        if torch.cuda.is_available():
            gpu_memory_used = torch.cuda.memory_allocated() / (1024**3)  # Convert to GB

        return {
            "cpu_percent": psutil.cpu_percent(),
            "ram_percent": psutil.virtual_memory().percent,
            "ram_used_gb": psutil.virtual_memory().used / (1024**3),
            "gpu_memory_used_gb": gpu_memory_used,
            "gpu_percent": torch.cuda.utilization() if torch.cuda.is_available() else 0,
        }

    def evaluate_model(
        self,
        model_name: str,
        model: Any,
        train_metrics: dict[str, Any],
        dataset_loader: Any,
        model_type: str = "segmentation",  # or "detection"
    ) -> dict[str, Any]:
        """
        Evaluate a model's performance and resource usage.

        Args:
            model_name: Name of the model
            model: The model object
            train_metrics: Training metrics
            dataset_loader: DataLoader for evaluation
            model_type: Type of model ("detection" or "segmentation")
        """
        print(f"\nEvaluating {model_name}...")

        # Get model size
        model_path = self.models_dir / f"{model_name.lower()}_model.pt"
        if hasattr(model, "save"):
            model.save(model_path)
        else:
            torch.save(model.state_dict(), model_path)
        model_size = model_path.stat().st_size / (1024 * 1024)  # MB

        # Evaluate inference performance
        inference_metrics = self.evaluate_inference(
            model, dataset_loader, model_type=model_type
        )

        # Get resource metrics during inference
        resource_metrics = self.get_resource_usage()

        # Combine all metrics
        self.results[model_name] = {
            "model_type": model_type,
            "model_size_mb": model_size,
            "training_metrics": train_metrics,
            "inference_metrics": inference_metrics,
            "resource_metrics": resource_metrics,
        }

        # Log to TensorBoard
        self._log_to_tensorboard(model_name, self.results[model_name])

        return self.results[model_name]

    def evaluate_inference(
        self,
        model: Any,
        dataloader: Any,
        num_iterations: int = 50,
        model_type: str = "detection",
    ) -> dict[str, float]:
        """Evaluate inference performance."""
        print("Evaluating inference performance...")
        inference_times = []
        batch_sizes = []
        accuracy_metrics = []

        with torch.no_grad():
            for _ in tqdm(range(num_iterations)):
                for batch in dataloader:
                    if model_type == "detection":
                        images = batch[0].to(self.device)
                        targets = batch[1]
                    else:  # segmentation
                        images, masks = batch
                        images = images.to(self.device)
                        masks = masks.to(self.device)

                    batch_sizes.append(images.shape[0])

                    # Measure inference time
                    start_time = time.time()
                    outputs = model(images)
                    inference_time = time.time() - start_time
                    inference_times.append(inference_time)

                    # Calculate accuracy metrics
                    if model_type == "detection":
                        # mAP calculation for detection models
                        accuracy = self._calculate_detection_metrics(outputs, targets)
                    else:
                        # IoU calculation for segmentation models
                        accuracy = self._calculate_segmentation_metrics(outputs, masks)
                    accuracy_metrics.append(accuracy)

        # Calculate statistics
        mean_time = np.mean(inference_times)
        return {
            "mean_inference_time": mean_time,
            "std_inference_time": np.std(inference_times),
            "fps": 1.0 / mean_time,
            "mean_batch_size": np.mean(batch_sizes),
            "mean_accuracy": np.mean(accuracy_metrics),
            "std_accuracy": np.std(accuracy_metrics),
        }

    def generate_report(self) -> Path:
        """Generate comprehensive evaluation report."""
        # Create DataFrame for results
        records = []
        for model_name, metrics in self.results.items():
            record = {
                "Model": model_name,
                "Type": metrics["model_type"],
                "Model Size (MB)": metrics["model_size_mb"],
                "Training Time (min)": metrics["training_metrics"]["time_minutes"],
                "Inference Speed (FPS)": metrics["inference_metrics"]["fps"],
                "Mean Accuracy": metrics["inference_metrics"]["mean_accuracy"],
                "GPU Memory (GB)": metrics["resource_metrics"]["gpu_memory_used_gb"],
                "RAM Usage (GB)": metrics["resource_metrics"]["ram_used_gb"],
            }
            records.append(record)

        df = pd.DataFrame(records)

        # Save detailed report
        report_path = self.results_dir / "comparison_report.csv"
        df.to_csv(report_path, index=False)

        # Create visualizations
        self._create_comparison_plots(df)

        return report_path

    def _create_comparison_plots(self, df: pd.DataFrame):
        """Create detailed comparison plots."""
        # Set style
        plt.style.use("seaborn")

        # Create subplots for different metrics
        fig = plt.figure(figsize=(20, 15))
        metrics = [
            ("Inference Speed (FPS)", "Inference Speed Comparison"),
            ("Model Size (MB)", "Model Size Comparison"),
            ("Training Time (min)", "Training Time Comparison"),
            ("Mean Accuracy", "Accuracy Comparison"),
            ("GPU Memory (GB)", "GPU Memory Usage"),
            ("RAM Usage (GB)", "RAM Usage"),
        ]

        for idx, (metric, title) in enumerate(metrics, 1):
            ax = fig.add_subplot(3, 2, idx)

            # Create grouped bar plots based on model type
            sns.barplot(data=df, x="Model", y=metric, hue="Type", ax=ax)

            ax.set_title(title)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

        plt.tight_layout()
        plt.savefig(
            self.plots_dir / "comparison_plots.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _log_to_tensorboard(self, model_name: str, metrics: dict[str, Any]):
        """Log metrics to TensorBoard."""
        # Log training metrics
        for metric_name, value in metrics["training_metrics"].items():
            self.writer.add_scalar(f"{model_name}/training/{metric_name}", value, 0)

        # Log inference metrics
        for metric_name, value in metrics["inference_metrics"].items():
            self.writer.add_scalar(f"{model_name}/inference/{metric_name}", value, 0)

        # Log resource metrics
        for metric_name, value in metrics["resource_metrics"].items():
            self.writer.add_scalar(f"{model_name}/resources/{metric_name}", value, 0)

    @staticmethod
    def _calculate_detection_metrics(outputs, targets) -> float:
        """Calculate mAP for detection models."""
        # Implementation depends on model output format
        # Return mAP score
        return 0.0  # Placeholder

    @staticmethod
    def _calculate_segmentation_metrics(outputs, masks) -> float:
        """Calculate IoU for segmentation models."""
        # Implementation depends on model output format
        # Return IoU score
        return 0.0  # Placeholder
