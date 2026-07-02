"""
main.py
-------
Entry point - just run:

    python main.py

Everything is configured through config.py. There are no command-line
flags; edit config.py to change paths or hyperparameters.
"""

import config
from preprocessor import load_dataset
from dataset import get_dataloaders
from model import UNet
from trainer import Trainer


def run_training():
    print(f"Loading dataset from {config.DATASET_ROOT} ...")
    images, labels = load_dataset()
    print(f"Loaded {len(images)} patches of size {config.PATCH_SIZE}x{config.PATCH_SIZE}.")

    train_loader, test_loader = get_dataloaders(images, labels)
    print(f"Train batches: {len(train_loader)}  Test batches: {len(test_loader)}")

    model = UNet()
    trainer = Trainer(model, train_loader, test_loader)
    trainer.train()

    print(f"Done. Checkpoints in {config.CHECKPOINT_DIR}, plots in {config.OUTPUT_DIR}.")


if __name__ == "__main__":
    run_training()
