import time

import detectron2
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from ultralytics import YOLO


class ModelTrainer:
    def __init__(self, config, dataset_manager):
        self.config = config
        self.dataset_manager = dataset_manager
        self.device = config["DEVICE"]

    def train_yolov8(self):
        print("\nTraining YOLOv8...")
        start_time = time.time()

        model = YOLO("yolov8n-seg.pt")
        results = model.train(
            data=self.dataset_manager.dataset_paths["yolo"]["data_yaml"],
            epochs=self.config["EPOCHS"],
            imgsz=self.config["IMAGE_SIZE"],
            device=self.device,
        )
        print(f"Training results: {results=}")

        model.save("saved_models/yolov8_model.pt")
        return {"model": model, "training_time": time.time() - start_time}

    def train_detectron2(self):
        print("\nTraining Detectron2...")
        start_time = time.time()

        cfg = get_cfg()
        cfg.merge_from_file(
            detectron2.model_zoo.get_config_file(
                "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
            )
        )
        cfg.DATASETS.TRAIN = (self.dataset_manager.dataset_paths["coco"]["train"],)
        cfg.DATASETS.TEST = (self.dataset_manager.dataset_paths["coco"]["valid"],)
        cfg.SOLVER.IMS_PER_BATCH = self.config["BATCH_SIZE"]
        cfg.SOLVER.MAX_ITER = self.config["EPOCHS"] * 100
        cfg.MODEL.DEVICE = self.device

        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()

        torch.save(trainer.model.state_dict(), "saved_models/detectron2_model.pth")
        return {"model": trainer.model, "training_time": time.time() - start_time}

    def train_bisenet(self):
        print("\nTraining BiSeNet...")
        model = smp.BiSeNet(
            encoder_name="resnet18", encoder_weights="imagenet", classes=1
        ).to(self.device)

        return self._train_generic_model(model, "bisenet")

    def train_deeplabv3(self):
        print("\nTraining DeepLabV3...")
        model = smp.DeepLabV3(
            encoder_name="resnet50", encoder_weights="imagenet", classes=1
        ).to(self.device)

        return self._train_generic_model(model, "deeplabv3")

    def _train_generic_model(self, model, name):
        start_time = time.time()
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.config["LEARNING_RATE"]
        )

        for epoch in range(self.config["EPOCHS"]):
            model.train()
            for batch_idx, (images, masks) in enumerate(
                self.dataset_manager.generic_dataloader
            ):
                images = images.to(self.device)
                masks = masks.to(self.device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

                if batch_idx % 10 == 0:
                    print(
                        f"Epoch {epoch+1}/{self.config['EPOCHS']}, "
                        f"Batch {batch_idx}, Loss: {loss.item():.4f}"
                    )

        torch.save(model.state_dict(), f"saved_models/{name}_model.pth")
        return {"model": model, "training_time": time.time() - start_time}
