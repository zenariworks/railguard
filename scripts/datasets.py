"""Manage download and upload of datasets."""
import os
from roboflow import Roboflow
from dotenv import load_dotenv

def download_dataset(workspace, project_name, version_number, api_key, output_format="yolov8", location="./data/"):
    """
    Downloads a dataset from Roboflow.

    Parameters:
        api_key (str): Your Roboflow API key.
        workspace (str): The name of your Roboflow workspace.
        project_name (str): The name of the project in Roboflow.
        version_number (int): The version number of the dataset to download.
        output_format (str): The format to download the dataset in (default is "yolov8").
        location (str): The directory where the dataset will be downloaded (default is './data/').

    Returns:
        dataset: The downloaded dataset object.
    """
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_name)
    version = project.version(version_number)
    
    # Ensure the download path exists
    os.makedirs(location, exist_ok=True)
    
    dataset = version.download(output_format, location=location)
    return dataset

if __name__ == "__main__":
    import argparse

    load_dotenv()

    parser = argparse.ArgumentParser(description="Download a dataset from Roboflow.")
    parser.add_argument("--api_key", type=str, help="Your Roboflow API key.")
    parser.add_argument("--workspace", type=str, required=True, help="The name of your Roboflow workspace.")
    parser.add_argument("--project", type=str, required=True, help="The name of the project in Roboflow.")
    parser.add_argument("--version", type=int, required=True, help="The version number of the dataset to download.")
    parser.add_argument("--format", type=str, default="yolov8", help="The format to download the dataset in (default is 'yolov8').")
    parser.add_argument("--location", type=str, default="./data/", help="The directory where the dataset will be downloaded (default is './data/').")
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("ROBOFLOW_API_KEY")

    if not api_key or not isinstance(api_key, str):
        raise ValueError("API key must be provided either via command line or as an environment variable and must be a string.")

    dataset = download_dataset(args.workspace, args.project, args.version, api_key, args.format, args.location)
    print(f"Dataset downloaded: {dataset}")