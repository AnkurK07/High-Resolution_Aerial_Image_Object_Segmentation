"""
preprocessor.py
---------------
Turns the raw tiled dataset (large aerial images + color-coded masks)
into fixed-size patches ready for training. Uses only OpenCV / numpy -
no TensorFlow, no patchify dependency.
"""

import os
import cv2
import numpy as np

import config


def _hex_to_rgb(hex_color: str) -> np.ndarray:
    hex_color = hex_color.lstrip("#")
    return np.array([int(hex_color[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.uint8)


def get_class_map() -> dict:
    """{class_name: rgb_array}, in the same order as config.CLASS_COLORS."""
    return {name: _hex_to_rgb(hex_val) for name, hex_val in config.CLASS_COLORS.items()}


def extract_patches(image: np.ndarray, patch_size: int) -> np.ndarray:
    """Split an image into non-overlapping square patches.

    Returns shape (num_patches, patch_size, patch_size, channels).
    Any border that doesn't fill a whole patch is discarded, same as the
    original notebook's crop-then-patchify step.
    """
    h, w = image.shape[0], image.shape[1]
    h_crop = (h // patch_size) * patch_size
    w_crop = (w // patch_size) * patch_size
    image = image[:h_crop, :w_crop]

    patches = []
    for y in range(0, h_crop, patch_size):
        for x in range(0, w_crop, patch_size):
            patches.append(image[y:y + patch_size, x:x + patch_size])
    return np.array(patches)


def rgb_mask_to_label(mask: np.ndarray, class_map: dict) -> np.ndarray:
    """Convert an (H, W, 3) color-coded mask into an (H, W) integer label map."""
    label = np.zeros(mask.shape[:2], dtype=np.uint8)
    for class_id, rgb in enumerate(class_map.values()):
        label[np.all(mask == rgb, axis=-1)] = class_id
    return label


def _iter_dataset_files():
    """Yield (image_path, mask_path) for every tile/image pair that exists."""
    for tile_id in range(1, config.NUM_TILES + 1):
        for image_id in range(1, config.IMAGES_PER_TILE + 1):
            name = f"image_part_{image_id:03d}"
            image_path = os.path.join(
                config.DATASET_ROOT, f"Tile {tile_id}", config.IMAGE_SUBDIR,
                f"{name}.{config.IMAGE_EXT}",
            )
            mask_path = os.path.join(
                config.DATASET_ROOT, f"Tile {tile_id}", config.MASK_SUBDIR,
                f"{name}.{config.MASK_EXT}",
            )
            if os.path.exists(image_path) and os.path.exists(mask_path):
                yield image_path, mask_path


def load_dataset():
    """Read every tile, patchify it, and return (images, labels) as numpy arrays.

    images: float32, shape (N, patch, patch, 3), normalized to [0, 1]
    labels: int64,   shape (N, patch, patch), class index per pixel
    """
    class_map = get_class_map()
    images, labels = [], []

    for image_path, mask_path in _iter_dataset_files():
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2RGB)

        for patch in extract_patches(image, config.PATCH_SIZE):
            images.append(patch.astype(np.float32) / 255.0)
        for patch in extract_patches(mask, config.PATCH_SIZE):
            labels.append(rgb_mask_to_label(patch, class_map))

    if not images:
        raise RuntimeError(
            f"No image/mask pairs found under {config.DATASET_ROOT}. "
            "Check config.DATASET_ROOT and the 'Tile N/images|masks' folder names."
        )

    return np.array(images), np.array(labels, dtype=np.int64)
