"""
trainer.py
----------
Training loop, validation loop, checkpointing, and loss/IoU curve plots.
"""

import os
import torch
import matplotlib.pyplot as plt

import config
from losses import CombinedLoss, iou_score


class Trainer:
    def __init__(self, model, train_loader, test_loader):
        self.device = torch.device(config.DEVICE)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.criterion = CombinedLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)

        self.history = {"loss": [], "val_loss": [], "iou": [], "val_iou": []}
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    def _run_epoch(self, loader, train: bool):
        self.model.train(train)
        total_loss, total_iou, n_batches = 0.0, 0.0, 0

        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)

            with torch.set_grad_enabled(train):
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            total_loss += loss.item()
            total_iou += iou_score(logits, labels)
            n_batches += 1

        return total_loss / n_batches, total_iou / n_batches

    def train(self, num_epochs=None):
        num_epochs = num_epochs or config.NUM_EPOCHS
        best_val_loss = float("inf")

        for epoch in range(1, num_epochs + 1):
            train_loss, train_iou = self._run_epoch(self.train_loader, train=True)
            val_loss, val_iou = self._run_epoch(self.test_loader, train=False)

            self.history["loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["iou"].append(train_iou)
            self.history["val_iou"].append(val_iou)

            print(f"Epoch {epoch}/{num_epochs} - "
                  f"loss: {train_loss:.4f} - iou: {train_iou:.4f} - "
                  f"val_loss: {val_loss:.4f} - val_iou: {val_iou:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(),
                           os.path.join(config.CHECKPOINT_DIR, "best_model.pt"))

        torch.save(self.model.state_dict(),
                   os.path.join(config.CHECKPOINT_DIR, "last_model.pt"))
        self.plot_history()

    def plot_history(self):
        epochs = range(1, len(self.history["loss"]) + 1)

        plt.figure(figsize=(14, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.history["loss"], label="Training Loss")
        plt.plot(epochs, self.history["val_loss"], label="Validation Loss")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.title("Loss")

        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.history["iou"], label="Training IoU")
        plt.plot(epochs, self.history["val_iou"], label="Validation IoU")
        plt.xlabel("Epoch"); plt.ylabel("IoU"); plt.legend(); plt.title("IoU")

        plt.tight_layout()
        plt.savefig(os.path.join(config.OUTPUT_DIR, "training_curves.png"))
        plt.close()
