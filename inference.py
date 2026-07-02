"""
inference.py
------------
Load a trained checkpoint and run prediction on a single image, with a
side-by-side visualization (same idea as the notebook's Gradio demo,
minus the web UI - call these functions from a notebook cell or script
instead).
"""

import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

import config
from model import UNet


def load_model(checkpoint_path: str = None) -> UNet:
    checkpoint_path = checkpoint_path or os.path.join(config.CHECKPOINT_DIR, "best_model.pt")
    model = UNet()
    model.load_state_dict(torch.load(checkpoint_path, map_location=config.DEVICE))
    model.to(config.DEVICE)
    model.eval()
    return model


def predict_image(model: UNet, image_path: str):
    """Run the model on one image file. Returns (resized_rgb_image, predicted_label_map)."""
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (config.PATCH_SIZE, config.PATCH_SIZE))

    tensor = torch.from_numpy(image.astype(np.float32) / 255.0)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(config.DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        prediction = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

    return image, prediction


def visualize(image: np.ndarray, prediction: np.ndarray, save_path: str = None):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image)
    plt.subplot(1, 2, 2)
    plt.title("Predicted Mask")
    plt.imshow(prediction)
    if save_path:
        plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    model = load_model()
    example_image_path = "path/to/some_image.jpg"   # <-- change this
    img, pred = predict_image(model, example_image_path)
    visualize(img, pred, save_path=os.path.join(config.OUTPUT_DIR, "prediction.png"))
