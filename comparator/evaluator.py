import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import torch
from tqdm import tqdm


class Evaluator:
    def __init__(self, config):
        self.config = config
        self.device = config["DEVICE"]
        self.results = {}

    def get_resource_usage(self):
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_used_gb": psutil.virtual_memory().used / (1024**3),
        }

    def evaluate_model(self, model_name, model_data, dataset_manager):
        model = model_data["model"]
        training_time = model_data["training_time"]

        # Get model size
        model_size = os.path.getsize(f"saved_models/{model_name.lower()}_model.p*") / (
            1024 * 1024
        )

        # Evaluate inference speed
        inference_speed = self.evaluate_inference_speed(model, dataset_manager)

        self.results[model_name] = {
            "training_time": training_time,
            "model_size_mb": model_size,
            "inference_speed": inference_speed,
        }

    def evaluate_inference_speed(self, model, dataset_manager, num_iterations=50):
        print("Evaluating inference speed...")
        inference_times = []

        with torch.no_grad():
            for _ in tqdm(range(num_iterations)):
                for images, _ in dataset_manager.generic_dataloader:
                    images = images.to(self.device)
                    start_time = time.time()
                    _ = model(images)
                    inference_times.append(time.time() - start_time)

        return {
            "mean_inference_time": np.mean(inference_times),
            "std_inference_time": np.std(inference_times),
            "fps": 1.0 / np.mean(inference_times),
        }

    def generate_report(self):
        df = pd.DataFrame()

        for model_name, metrics in self.results.items():
            df = df.append(
                {
                    "Model": model_name,
                    "Training Time (min)": metrics["training_time"] / 60,
                    "Inference Speed (FPS)": metrics["inference_speed"]["fps"],
                    "Model Size (MB)": metrics["model_size_mb"],
                },
                ignore_index=True,
            )

        # Save report
        report_path = "results/comparison_report.csv"
        df.to_csv(report_path, index=False)

        # Create visualizations
        self.create_comparison_plots(df)

        return report_path

    def create_comparison_plots(self, df):
        plt.figure(figsize=(15, 10))

        metrics = [
            ("Inference Speed (FPS)", "Inference Speed Comparison"),
            ("Model Size (MB)", "Model Size Comparison"),
            ("Training Time (min)", "Training Time Comparison"),
        ]

        for idx, (metric, title) in enumerate(metrics, 1):
            plt.subplot(2, 2, idx)
            plt.bar(df["Model"], df[metric])
            plt.title(title)
            plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig("results/comparison_plots.png")
        plt.close()
