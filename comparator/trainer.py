import json
import os
import time
from datetime import datetime
from pathlib import Path

import detectron2
import matplotlib.pyplot as plt
import psutil
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from ultralytics import YOLO


class RailTrackModelComparison:
    def __init__(self, config: dict):
        self.config = config
        # Ensure we're using MPS if available, otherwise CPU
        self.device = config["DEVICE"]
        print(f"Using device: {self.device}")

        # Get the current working directory and construct paths relative to it
        self.cwd = Path.cwd()
        self.data_dir = self.cwd / "data"
        self.version_dir = self.data_dir / f"version_{config['VERSION']}"
        self.output_dir = self.cwd / "output"
        self.output_dir.mkdir(exist_ok=True)

        print(f"Working directory: {self.cwd}")
        print(f"Data directory: {self.data_dir}")
        print(f"Version directory: {self.version_dir}")

        # Validate paths
        if not self.version_dir.exists():
            raise FileNotFoundError(f"Version directory not found: {self.version_dir}")

        # Initialize metrics storage
        self.metrics = {
            "yolov8": {"train": {}, "inference": {}, "predictions": []},
            "detectron2": {"train": {}, "inference": {}, "predictions": []},
        }

    def _measure_resources(self, start_time: float, start_memory: float) -> dict:
        """Measure computational resources specific to MacOS."""
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        return {
            "time_seconds": end_time - start_time,
            "memory_mb": end_memory - start_memory,
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
        }

    def train_yolov8(self) -> dict:
        """Train YOLOv8 model for rail track detection."""
        print("\nTraining YOLOv8...")
        yolo_dir = self.version_dir / "yolov8"

        # Verify YOLO directory structure
        required_dirs = [
            yolo_dir / "train" / "images",
            yolo_dir / "valid" / "images",
            yolo_dir / "test" / "images",
        ]

        for dir_path in required_dirs:
            if not dir_path.exists():
                raise FileNotFoundError(f"Required directory not found: {dir_path}")

        # Create data.yaml in the yolo_dir if it doesn't exist
        data_yaml = yolo_dir / "data.yaml"
        if not data_yaml.exists():
            yaml_content = {
                "path": str(self.version_dir / "yolov8"),  # Use relative path
                "train": str(yolo_dir / "train"),
                "val": str(yolo_dir / "valid"),
                "test": str(yolo_dir / "test"),
                "names": {0: "rail_track"},
                "nc": 1,
            }
            import yaml

            with open(data_yaml, "w") as f:
                yaml.safe_dump(yaml_content, f)

        # Initialize model with nano size for efficiency on MPS
        model = YOLO("yolov8n.pt")

        # Measure training resources
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024

        # Train model with verified paths
        results = model.train(
            data=str(data_yaml),
            epochs=self.config["EPOCHS"],
            imgsz=self.config["IMAGE_SIZE"],
            batch=self.config["BATCH_SIZE"],
            device=self.device,
            project=str(self.output_dir).replace("/", "_"),
            name="yolov8_rail_tracks",
            save=True,
        )

        # Record metrics
        self.metrics["yolov8"]["train"] = {
            "resources": self._measure_resources(start_time, start_memory),
            "mAP50": float(results.results_dict["metrics/mAP50(B)"]),
            "mAP50-95": float(results.results_dict["metrics/mAP50-95(B)"]),
            "precision": float(results.results_dict["metrics/precision(B)"]),
            "recall": float(results.results_dict["metrics/recall(B)"]),
        }

        return self.metrics["yolov8"]["train"]

    def train_detectron2(self) -> dict:
        """Train Detectron2 model for rail track detection."""
        print("\nTraining Detectron2...")
        coco_dir = self.version_dir / "coco"

        # Register datasets
        register_coco_instances(
            "rail_tracks_train",
            {},
            str(coco_dir / "train/_annotations.coco.json"),
            str(coco_dir / "train"),
        )
        register_coco_instances(
            "rail_tracks_val",
            {},
            str(coco_dir / "valid/_annotations.coco.json"),
            str(coco_dir / "valid"),
        )

        # Configure model
        cfg = get_cfg()
        cfg.merge_from_file(
            detectron2.model_zoo.get_config_file(
                "COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"  # Using 1x schedule for faster training
            )
        )
        cfg.DATASETS.TRAIN = ("rail_tracks_train",)
        cfg.DATASETS.TEST = ("rail_tracks_val",)
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Only rail tracks
        cfg.MODEL.DEVICE = str(self.device)
        cfg.SOLVER.IMS_PER_BATCH = self.config["BATCH_SIZE"]
        cfg.SOLVER.MAX_ITER = self.config["EPOCHS"] * 100
        cfg.SOLVER.BASE_LR = self.config["LEARNING_RATE"]
        cfg.OUTPUT_DIR = str(self.output_dir / "detectron2_rail_tracks")
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

        # Measure training resources
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024

        # Train model
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()

        # Evaluate
        evaluator = COCOEvaluator(
            "rail_tracks_val", cfg, False, output_dir=cfg.OUTPUT_DIR
        )
        results = DefaultTrainer.test(cfg, trainer.model, evaluators=[evaluator])

        # Record metrics
        self.metrics["detectron2"]["train"] = {
            "resources": self._measure_resources(start_time, start_memory),
            "mAP": results["bbox"]["AP"],
            "AP50": results["bbox"]["AP50"],
            "AP75": results["bbox"]["AP75"],
        }

        return self.metrics["detectron2"]["train"]

    def benchmark_inference(self, num_images: int = 20) -> dict:
        """Benchmark inference performance on test images."""
        print("\nBenchmarking inference...")
        test_images = list((self.version_dir / "yolov8/test/images").glob("*.jpg"))[
            :num_images
        ]

        results = {}
        for model_name in ["yolov8", "detectron2"]:
            print(f"\nTesting {model_name}...")

            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024

            if model_name == "yolov8":
                model = YOLO(
                    str(self.output_dir / "yolov8_rail_tracks/weights/best.pt")
                )
                for img_path in test_images:
                    prediction = model(img_path)
                    self.metrics[model_name]["predictions"].append(
                        {
                            "image": str(img_path),
                            "boxes": prediction[0].boxes.data.cpu().numpy().tolist(),
                        }
                    )
            else:
                cfg = get_cfg()
                cfg.merge_from_file(
                    str(self.output_dir / "detectron2_rail_tracks/config.yml")
                )
                predictor = DefaultPredictor(cfg)
                for img_path in test_images:
                    img = detectron2.data.detection_utils.read_image(str(img_path))
                    prediction = predictor(img)
                    self.metrics[model_name]["predictions"].append(
                        {
                            "image": str(img_path),
                            "boxes": prediction["instances"]
                            .pred_boxes.tensor.cpu()
                            .numpy()
                            .tolist(),
                        }
                    )

            results[model_name] = self._measure_resources(start_time, start_memory)
            results[model_name]["images_per_second"] = (
                num_images / results[model_name]["time_seconds"]
            )

        self.metrics["inference_comparison"] = results
        return results

    def visualize_results(self):
        """Create visualizations comparing model performance."""
        # Create comparison plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Training time comparison
        times = [
            self.metrics["yolov8"]["train"]["resources"]["time_seconds"],
            self.metrics["detectron2"]["train"]["resources"]["time_seconds"],
        ]
        ax1.bar(["YOLOv8", "Detectron2"], times)
        ax1.set_title("Training Time (seconds)")
        ax1.set_ylabel("Seconds")

        # Memory usage comparison
        memory = [
            self.metrics["yolov8"]["train"]["resources"]["memory_mb"],
            self.metrics["detectron2"]["train"]["resources"]["memory_mb"],
        ]
        ax2.bar(["YOLOv8", "Detectron2"], memory)
        ax2.set_title("Memory Usage (MB)")
        ax2.set_ylabel("MB")

        # Inference speed comparison
        inference_speed = [
            self.metrics["inference_comparison"]["yolov8"]["images_per_second"],
            self.metrics["inference_comparison"]["detectron2"]["images_per_second"],
        ]
        ax3.bar(["YOLOv8", "Detectron2"], inference_speed)
        ax3.set_title("Inference Speed (images/second)")
        ax3.set_ylabel("Images per second")

        # Accuracy comparison (mAP50)
        accuracy = [
            self.metrics["yolov8"]["train"]["mAP50"],
            self.metrics["detectron2"]["train"]["AP50"],
        ]
        ax4.bar(["YOLOv8", "Detectron2"], accuracy)
        ax4.set_title("Accuracy (mAP50)")
        ax4.set_ylabel("mAP50")

        plt.tight_layout()
        plt.savefig(self.output_dir / "model_comparison.png")
        plt.close()

    def save_results(self):
        """Save detailed comparison results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = (
            self.output_dir / f"rail_track_detection_comparison_{timestamp}.json"
        )

        with open(results_path, "w") as f:
            json.dump(self.metrics, f, indent=4)

        print(f"\nResults saved to {results_path}")

    def run_comparison(self):
        """Run complete comparison pipeline."""
        print("Starting model comparison for rail track detection...")

        # Train models
        yolo_metrics = self.train_yolov8()
        detectron_metrics = self.train_detectron2()

        # Benchmark inference
        inference_metrics = self.benchmark_inference()

        # Create visualizations
        self.visualize_results()

        # Save results
        self.save_results()

        # Print summary
        print("\n=== Comparison Summary ===")
        print("\nTraining Metrics:")
        print("YOLOv8:")
        print(f"- Time: {yolo_metrics['resources']['time_seconds']:.2f}s")
        print(f"- mAP50: {yolo_metrics['mAP50']:.3f}")
        print(f"- Memory: {yolo_metrics['resources']['memory_mb']:.1f}MB")

        print("\nDetectron2:")
        print(f"- Time: {detectron_metrics['resources']['time_seconds']:.2f}s")
        print(f"- AP50: {detectron_metrics['AP50']:.3f}")
        print(f"- Memory: {detectron_metrics['resources']['memory_mb']:.1f}MB")

        print("\nInference Metrics:")
        print("YOLOv8:")
        print(
            f"- Speed: {inference_metrics['yolov8']['images_per_second']:.2f} images/second"
        )
        print(f"- Memory: {inference_metrics['yolov8']['memory_mb']:.1f}MB")

        print("\nDetectron2:")
        print(
            f"- Speed: {inference_metrics['detectron2']['images_per_second']:.2f} images/second"
        )
        print(f"- Memory: {inference_metrics['detectron2']['memory_mb']:.1f}MB")
