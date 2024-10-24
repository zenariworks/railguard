"""Trainer module for comparing different object detection models with Colab support."""

import json
import os
import time
from pathlib import Path
from typing import Dict, Optional, Union

import torch
import yaml
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from ultralytics import YOLO


class RailTrackModelComparison:
    """Handles training and comparison of different object detection models."""
    
    def __init__(self, config: dict, use_drive: bool = False):
        """
        Initialize the trainer with configuration.
        
        Args:
            config (dict): Configuration dictionary
            use_drive (bool): Whether to use Google Drive for storage
        """
        self.config = config
        self.use_drive = use_drive
        
        if self.use_drive:
            try:
                from google.colab import drive
                drive.mount('/content/drive')
                self.base_path = Path("/content/drive/MyDrive/rail_track_comparison")
            except ImportError:
                print("Google Colab import failed. Falling back to local storage.")
                self.use_drive = False
                self.base_path = Path("results")
        else:
            self.base_path = Path("results")
            
        # Create necessary directories
        self.results_dir = self.base_path / "results"
        self.model_dir = self.base_path / "models"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.results_file = self.results_dir / "comparison_results.json"
        self.checkpoint_file = self.results_dir / "checkpoint.json"
        self.results = self._load_previous_results()
        
    def _load_previous_results(self) -> dict:
        """Load previous results if they exist."""
        if self.results_file.exists():
            with open(self.results_file, "r") as f:
                return json.load(f)
        return {
            "yolov8": {},
            "detectron2": {},
            "training_status": {
                "yolov8": False,
                "detectron2": False
            }
        }
    
    def _save_results(self):
        """Save current results to file."""
        with open(self.results_file, "w") as f:
            json.dump(self.results, f, indent=4)
            
    def _save_checkpoint(self, model_type: str):
        """Save checkpoint for the current model."""
        with open(self.checkpoint_file, "w") as f:
            json.dump({
                "last_completed": model_type,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }, f)
            
    def _get_model_save_path(self, model_type: str) -> Path:
        """Get path for saving model weights."""
        return self.model_dir / f"{model_type}_weights"
            
    def train_yolov8(self, data_path: str):
        """Train YOLOv8 model if not already trained."""
        if self.results["training_status"]["yolov8"]:
            print("YOLOv8 model already trained. Skipping...")
            return
        
        print("Training YOLOv8 model...")
        start_time = time.time()
        
        # Initialize model
        model = YOLO("yolov8n.pt")
        
        # Update data.yaml path
        data_yaml = Path(data_path) / "data.yaml"
        with open(data_yaml, "r") as f:
            data_config = yaml.safe_load(f)
        
        # Set up save directory
        save_dir = self._get_model_save_path("yolov8")
        
        # Train model
        results = model.train(
            data=str(data_yaml),
            epochs=self.config["EPOCHS"],
            imgsz=self.config["IMAGE_SIZE"],
            batch=self.config["BATCH_SIZE"],
            device=self.config["DEVICE"],
            workers=self.config["NUM_WORKERS"],
            project=str(save_dir),
            name="train"
        )
        
        training_time = time.time() - start_time
        
        # Save metrics
        metrics = {
            "training_time": training_time,
            "final_map": float(results.results_dict["metrics/mAP50-95(B)"][-1]),
            "memory_usage": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
            "model_path": str(save_dir / "train" / "weights" / "best.pt")
        }
        
        self.results["yolov8"] = metrics
        self.results["training_status"]["yolov8"] = True
        self._save_results()
        self._save_checkpoint("yolov8")
        
    def train_detectron2(self, data_path: str):
        """Train Detectron2 model if not already trained."""
        if self.results["training_status"]["detectron2"]:
            print("Detectron2 model already trained. Skipping...")
            return
            
        print("Training Detectron2 model...")
        start_time = time.time()
        
        # Register dataset
        register_coco_instances(
            "rail_train",
            {},
            str(Path(data_path) / "train/_annotations.coco.json"),
            str(Path(data_path) / "train")
        )
        
        # Configure model
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg.DATASETS.TRAIN = ("rail_train",)
        cfg.DATASETS.TEST = ()
        cfg.DATALOADER.NUM_WORKERS = self.config["NUM_WORKERS"]
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Only one class (rail track)
        cfg.MODEL.DEVICE = self.config["DEVICE"]
        cfg.SOLVER.IMS_PER_BATCH = self.config["BATCH_SIZE"]
        cfg.SOLVER.BASE_LR = self.config["LEARNING_RATE"]
        cfg.SOLVER.MAX_ITER = self.config["EPOCHS"] * 100  # Approximate iterations per epoch
        
        # Set up save directory
        save_dir = self._get_model_save_path("detectron2")
        cfg.OUTPUT_DIR = str(save_dir)
        
        # Train model
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()
        
        training_time = time.time() - start_time
        
        # Save metrics
        metrics = {
            "training_time": training_time,
            "memory_usage": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
            "model_path": str(save_dir / "model_final.pth")
        }
        
        self.results["detectron2"] = metrics
        self.results["training_status"]["detectron2"] = True
        self._save_results()
        self._save_checkpoint("detectron2")
        
    def compare_models(self) -> Dict:
        """Compare trained models and return results."""
        if not all(self.results["training_status"].values()):
            missing = [k for k, v in self.results["training_status"].items() if not v]
            raise ValueError(f"Not all models have been trained. Missing: {missing}")
            
        comparison = {
            "training_time_comparison": {
                "yolov8": self.results["yolov8"]["training_time"],
                "detectron2": self.results["detectron2"]["training_time"],
            },
            "memory_usage_comparison": {
                "yolov8": self.results["yolov8"]["memory_usage"],
                "detectron2": self.results["detectron2"]["memory_usage"],
            },
            "model_paths": {
                "yolov8": self.results["yolov8"]["model_path"],
                "detectron2": self.results["detectron2"]["model_path"],
            }
        }
        
        if "final_map" in self.results["yolov8"]:
            comparison["yolov8_map"] = self.results["yolov8"]["final_map"]
            
        return comparison
        
    def save_to_drive(self, local_results_path: Optional[Path] = None):
        """
        Save results to Google Drive if available.
        
        Args:
            local_results_path: Optional path to local results to copy to Drive
        """
        if not self.use_drive:
            print("Google Drive not available. Results saved locally.")
            return
            
        # If a local path is provided, copy those results to Drive
        if local_results_path is not None and local_results_path.exists():
            import shutil
            drive_path = self.base_path / local_results_path.name
            shutil.copy2(local_results_path, drive_path)
            print(f"Copied local results to Drive: {drive_path}")