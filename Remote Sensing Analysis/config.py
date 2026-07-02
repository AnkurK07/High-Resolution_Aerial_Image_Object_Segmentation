import torch

# Paths 
DATASET_ROOT = "/path/to/Dataset"         

CHECKPOINT_DIR = "./checkpoints"           # trained model weights go here
OUTPUT_DIR = "./outputs"                   # plots / predictions go here

# Just Something for my custom datasets 
NUM_TILES = 7               
IMAGES_PER_TILE = 19        
IMAGE_SUBDIR = "images"
MASK_SUBDIR = "masks"
IMAGE_EXT = "jpg"
MASK_EXT = "png"

PATCH_SIZE = 256             # square patch size fed to the network

# RGB color for every class as it appears in the color-coded masks
CLASS_COLORS = {
    "water":      "#E2A929",
    "land":       "#8429F6",
    "road":       "#6EC1E4",
    "building":   "#3C1098",
    "vegetation": "#FEDD3A",
    "unlabeled":  "#9B9B9B",
}
NUM_CLASSES = len(CLASS_COLORS)


# Training hyperparameters
BATCH_SIZE = 16
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
TEST_SPLIT = 0.15
RANDOM_SEED = 100

BASE_FILTERS = 16          
DROPOUT = 0.2
IN_CHANNELS = 3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Loss settings (Dice + Focal, matching the original notebook's total loss)
FOCAL_GAMMA = 2.0
DICE_SMOOTH = 1.0
