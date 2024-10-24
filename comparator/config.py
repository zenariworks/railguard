import os

import torch
from dotenv import load_dotenv

load_dotenv(override=True)

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

CONFIG = {
    "WORKSPACE": "jet-znmu4",
    "PROJECT": "help_me",
    "DATA_DIR": "data",
    "VERSION": 5,
    "API_KEY": ROBOFLOW_API_KEY,
    "DEVICE": DEVICE,
    "IMAGE_SIZE": 640,
    "BATCH_SIZE": 4,
    "NUM_WORKERS": 2,
    "EPOCHS": 10,
    "LEARNING_RATE": 0.001,
}
