"""Module for managing datasets and data loading for rail track image processing."""

import os

from PIL import Image
from roboflow import Roboflow
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class RailTrackDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(
            [f for f in os.listdir(image_dir) if f.endswith((".jpg", ".png", ".jpeg"))]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace(".jpg", ".png"))

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = transforms.ToTensor()(mask)

        return image, mask


class DatasetManager:
    def __init__(self, config):
        self.config = config
        self.rf = Roboflow(api_key=config["API_KEY"])
        self.workspace = self.rf.workspace(config["WORKSPACE"])
        self.project = self.workspace.project(config["PROJECT"])

        # Create necessary directories
        self.create_directories()

        self.transform = transforms.Compose(
            [
                transforms.Resize((config["IMAGE_SIZE"], config["IMAGE_SIZE"])),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.prepare_datasets()

    def create_directories(self):
        """Create all necessary directories for the project"""
        dirs = [
            self.config["DATA_DIR"],
            os.path.join(self.config["DATA_DIR"], "yolo"),
            os.path.join(self.config["DATA_DIR"], "coco"),
            os.path.join(self.config["DATA_DIR"], "semantic"),
            "saved_models",
            "results",
        ]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)

    def prepare_datasets(self):
        print("Downloading and preparing datasets...")

        # Download datasets into specific directories
        yolo_path = os.path.join(self.config["DATA_DIR"], "yolo")
        coco_path = os.path.join(self.config["DATA_DIR"], "coco")
        semantic_path = os.path.join(self.config["DATA_DIR"], "semantic")

        self.yolo_dataset = self.project.version(self.config["VERSION"]).download(
            "yolov8", location=yolo_path
        )
        self.coco_dataset = self.project.version(self.config["VERSION"]).download("coco", location=coco_path)
        self.seg_dataset = self.project.version(self.config["VERSION"]).download(
            "semantic-segmentation", location=semantic_path
        )

        # Setup dataloaders
        self.setup_dataloaders()

    def setup_dataloaders(self):
        semantic_dir = os.path.join(self.config["DATA_DIR"], "semantic")
        self.generic_dataset = RailTrackDataset(
            image_dir=os.path.join(semantic_dir, "train/images"),
            mask_dir=os.path.join(semantic_dir, "train/masks"),
            transform=self.transform,
        )

        self.generic_dataloader = DataLoader(
            self.generic_dataset,
            batch_size=self.config["BATCH_SIZE"],
            shuffle=True,
            num_workers=self.config["NUM_WORKERS"],
        )

        # Update dataset paths with new directory structure
        self.dataset_paths = {
            "yolo": {
                "data_yaml": os.path.join(self.config["DATA_DIR"], "yolo/data.yaml"),
                "train_images": os.path.join(
                    self.config["DATA_DIR"], "yolo/train/images"
                ),
                "valid_images": os.path.join(
                    self.config["DATA_DIR"], "yolo/valid/images"
                ),
                "test_images": os.path.join(
                    self.config["DATA_DIR"], "yolo/test/images"
                ),
            },
            "coco": {
                "train": os.path.join(self.config["DATA_DIR"], "coco/train"),
                "valid": os.path.join(self.config["DATA_DIR"], "coco/valid"),
                "test": os.path.join(self.config["DATA_DIR"], "coco/test"),
            },
            "semantic": {
                "train_images": os.path.join(
                    self.config["DATA_DIR"], "semantic/train/images"
                ),
                "train_masks": os.path.join(
                    self.config["DATA_DIR"], "semantic/train/masks"
                ),
                "valid_images": os.path.join(
                    self.config["DATA_DIR"], "semantic/valid/images"
                ),
                "valid_masks": os.path.join(
                    self.config["DATA_DIR"], "semantic/valid/masks"
                ),
            },
        }
