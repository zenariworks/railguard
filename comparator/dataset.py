"""Module for managing datasets and data loading for rail track image processing."""
from pathlib import Path
from typing import Literal
from roboflow import Roboflow

class DatasetManager:
    """A simplified class to manage dataset downloads from Roboflow."""
    
    def __init__(self, config: dict):
        """
        Initialize the DatasetManager with configuration.
        
        Args:
            config (dict): Configuration dictionary containing necessary parameters
        """
        self.config = config
        self.data_dir = Path(config["DATA_DIR"])
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize Roboflow
        self.rf = Roboflow(api_key=config["API_KEY"])
        
    def download_dataset(self, data_format: Literal["yolov8", "coco"] = "yolov8") -> str:
        """
        Download dataset from Roboflow in specified format.
        
        Args:
            data_format: Format to download ("yolov8" or "coco")
            
        Returns:
            str: Path to the downloaded dataset
        """
        # Get project and version
        project = self.rf.workspace(self.config["WORKSPACE"]).project(self.config["PROJECT"])
        version = project.version(self.config["VERSION"])
        
        # Create directory for specific version
        version_dir = self.data_dir / f"version_{self.config['VERSION']}"
        version_dir.mkdir(exist_ok=True)
        
        # Download dataset
        download_path = version_dir / data_format
        if not download_path.exists():
            _ = version.download(
                model_format=data_format,
                location=str(download_path)
            )
            
        return str(download_path)
