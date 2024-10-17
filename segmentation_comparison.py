import torch
import time
import numpy as np
from roboflow import Roboflow
from ultralytics import YOLO
import psutil
from datetime import datetime
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn
from torchvision import models
import detectron2
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
import segmentation_models_pytorch as smp
from PIL import Image
import os
import pandas as pd

class SegmentationEvaluator:
    def __init__(self, workspace, project_name):
        self.rf = Roboflow(api_key="YOUR_API_KEY")
        self.workspace = self.rf.workspace(workspace)
        self.project = self.workspace.project(project_name)
        self.results = {}
        
        # Set device to MPS for MacOS
        self.device = (
            "mps" 
            if torch.backends.mps.is_available() 
            else "cpu"
        )
        print(f"Using device: {self.device}")
        
        # Create directories for saved models and results
        os.makedirs("saved_models", exist_ok=True)
        os.makedirs("results", exist_ok=True)
        
    def get_resource_usage(self):
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_used_gb": psutil.virtual_memory().used / (1024**3)
        }

    def train_yolov8(self, epochs=10):
        print("\nTraining YOLOv8...")
        dataset = self.project.version(self.config["VERSION"]).download("yolov8")
        
        start_time = time.time()
        initial_resources = self.get_resource_usage()
        
        model = YOLO("yolov8n-seg.pt")
        results = model.train(
            data=f"{dataset.location}/data.yaml",
            epochs=epochs,
            imgsz=640,
            device=self.device
        )
        
        model.save(f"saved_models/yolov8_model.pt")
        
        return self._calculate_metrics("YOLOv8", model, start_time, initial_resources)

    def train_detectron2(self, epochs=10):
        print("\nTraining Detectron2...")
        dataset = self.project.version(self.config["VERSION"]).download("coco")
        
        start_time = time.time()
        initial_resources = self.get_resource_usage()
        
        cfg = get_cfg()
        cfg.merge_from_file(detectron2.model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        ))
        cfg.DATASETS.TRAIN = ("railway_tracks_train",)
        cfg.DATASETS.TEST = ("railway_tracks_val",)
        cfg.SOLVER.IMS_PER_BATCH = 2
        cfg.SOLVER.MAX_ITER = epochs * 100
        cfg.MODEL.DEVICE = self.device
        
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()
        
        torch.save(trainer.model.state_dict(), "saved_models/detectron2_model.pth")
        
        return self._calculate_metrics("Detectron2", trainer.model, start_time, initial_resources)

    def train_bisenet(self, epochs=10):
        print("\nTraining BiSeNet...")
        model = smp.BiSeNet(
            encoder_name="resnet18",
            encoder_weights="imagenet",
            classes=1
        ).to(self.device)
        
        start_time = time.time()
        initial_resources = self.get_resource_usage()
        
        # Training loop would go here
        torch.save(model.state_dict(), "saved_models/bisenet_model.pth")
        
        return self._calculate_metrics("BiSeNet", model, start_time, initial_resources)

    def train_deeplabv3(self, epochs=10):
        print("\nTraining DeepLabV3...")
        model = smp.DeepLabV3(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            classes=1
        ).to(self.device)
        
        start_time = time.time()
        initial_resources = self.get_resource_usage()
        
        # Training loop would go here
        torch.save(model.state_dict(), "saved_models/deeplabv3_model.pth")
        
        return self._calculate_metrics("DeepLabV3", model, start_time, initial_resources)

    def _calculate_metrics(self, model_name, model, start_time, initial_resources):
        training_time = time.time() - start_time
        final_resources = self.get_resource_usage()
        
        # Evaluate inference speed
        inference_speed = self.evaluate_inference_speed(model)
        
        metrics = {
            "training_time": training_time,
            "resource_usage": {
                "initial": initial_resources,
                "final": final_resources,
                "delta_memory_gb": final_resources["memory_used_gb"] - 
                                 initial_resources["memory_used_gb"]
            },
            "inference_speed": inference_speed,
            "model_size_mb": os.path.getsize(f"saved_models/{model_name.lower()}_model.pth") / (1024*1024)
        }
        
        self.results[model_name] = metrics
        return metrics

    def evaluate_inference_speed(self, model, num_iterations=50):
        print(f"Evaluating inference speed...")
        test_dataset = self.project.version(self.config["VERSION"]).download("yolov8")
        test_images = [f for f in os.listdir(f"{test_dataset.location}/test/images") 
                      if f.endswith(('.jpg', '.png'))]
        
        inference_times = []
        
        with torch.no_grad():
            for _ in tqdm(range(num_iterations)):
                for img_path in test_images:
                    start_time = time.time()
                    # Perform inference (implementation depends on model type)
                    inference_times.append(time.time() - start_time)
        
        return {
            "mean_inference_time": np.mean(inference_times),
            "std_inference_time": np.std(inference_times),
            "fps": 1.0 / np.mean(inference_times)
        }

    def generate_comparison_report(self):
        df = pd.DataFrame()
        
        for model_name, metrics in self.results.items():
            df = df.append({
                'Model': model_name,
                'Training Time (min)': metrics['training_time'] / 60,
                'Inference Speed (FPS)': metrics['inference_speed']['fps'],
                'Memory Usage (GB)': metrics['resource_usage']['delta_memory_gb'],
                'Model Size (MB)': metrics['model_size_mb']
            }, ignore_index=True)
        
        # Save detailed report
        report_path = "results/comparison_report.csv"
        df.to_csv(report_path, index=False)
        
        # Create visualizations
        plt.figure(figsize=(15, 10))
        
        # Speed comparison
        plt.subplot(2, 2, 1)
        plt.bar(df['Model'], df['Inference Speed (FPS)'])
        plt.title('Inference Speed Comparison')
        plt.xticks(rotation=45)
        
        # Memory usage comparison
        plt.subplot(2, 2, 2)
        plt.bar(df['Model'], df['Memory Usage (GB)'])
        plt.title('Memory Usage Comparison')
        plt.xticks(rotation=45)
        
        # Model size comparison
        plt.subplot(2, 2, 3)
        plt.bar(df['Model'], df['Model Size (MB)'])
        plt.title('Model Size Comparison')
        plt.xticks(rotation=45)
        
        # Training time comparison
        plt.subplot(2, 2, 4)
        plt.bar(df['Model'], df['Training Time (min)'])
        plt.title('Training Time Comparison')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('results/comparison_plots.png')
        plt.close()
        
        return report_path

def main():
    # Initialize evaluator
    evaluator = SegmentationEvaluator("jet-znmu4", "help_me")
    
    # Train and evaluate all models
    models_to_train = [
        evaluator.train_yolov8,
        evaluator.train_detectron2,
        evaluator.train_bisenet,
        evaluator.train_deeplabv3
    ]
    
    for train_func in models_to_train:
        try:
            train_func(epochs=10)
        except Exception as e:
            print(f"Error training {train_func.__name__}: {str(e)}")
    
    # Generate comparison report
    report_path = evaluator.generate_comparison_report()
    print(f"\nComparison report saved to: {report_path}")
    print("Visualization saved to: results/comparison_plots.png")

if __name__ == "__main__":
    main()