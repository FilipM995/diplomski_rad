import math
import os
import random
import shutil
from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as TF
from tqdm import tqdm
from torchvision import transforms


# Function to apply augmentations
def augment_image(image, mask, fov):
    # Convert to tensors
    image, mask, fov = (
        transforms.ToTensor()(image),
        transforms.ToTensor()(mask),
        transforms.ToTensor()(fov),
    )

    # Random vertical flipping
    if random.random() > 0.5:
        image = TF.vflip(image)
        mask = TF.vflip(mask)
        fov = TF.vflip(fov)

    # Random rotation
    if random.random() > 0.5:
        angle = random.randint(0, 45)
        image = TF.rotate(image, angle)
        mask = TF.rotate(mask, angle)
        fov = TF.rotate(fov, angle)

    # # Random stretch
    # if random.random() > 0.5:
    #     angle = random.randint(0, 45)
    #     image = TF.el

    # Convert back to numpy arrays
    image, mask, fov = (
        TF.to_pil_image(image),
        TF.to_pil_image(mask),
        TF.to_pil_image(fov),
    )
    return image, mask, fov


def augment_dataset(
    original_image_dir,
    original_mask_dir,
    original_fov_dir,
    augmented_image_dir,
    augmented_mask_dir,
    augmented_fov_dir,
    desired_augmented_count,
):
    # Define paths
    original_image_dir = "archive/DRIVE/training/images"
    original_mask_dir = "archive/DRIVE/training/1st_manual"
    original_fov_dir = "archive/DRIVE/training/mask"
    augmented_image_dir = "augment/images"
    augmented_mask_dir = "augment/1st_manual"
    augmented_fov_dir = "augment/mask"

    # Create directories if they don't exist
    os.makedirs(augmented_image_dir, exist_ok=True)
    os.makedirs(augmented_mask_dir, exist_ok=True)
    os.makedirs(augmented_fov_dir, exist_ok=True)

    # Clear the contents of the augmented directories
    for folder in [augmented_image_dir, augmented_mask_dir, augmented_fov_dir]:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)


    # Get the number of original images
    original_image_count = len(os.listdir(original_image_dir))

    # Calculate the number of augmentations per original image
    augmentations_per_image = math.ceil(desired_augmented_count / original_image_count)

    # Iterate over the original dataset
    for image_name in tqdm(os.listdir(original_image_dir), desc="Augmenting"):
        image_path = os.path.join(original_image_dir, image_name)
        mask_path = os.path.join(
            original_mask_dir, image_name.replace("training.tif", "manual1.gif")
        )
        fov_path = os.path.join(
            original_fov_dir, image_name.replace(".tif", "_mask.gif")
        )

        # Load image and mask
        image = Image.open(image_path)
        mask = Image.open(mask_path)
        fov = Image.open(fov_path)

        # Save original image, mask, and fov
        original_image_path = os.path.join(augmented_image_dir, image_name)
        original_mask_path = os.path.join(
            augmented_mask_dir, image_name.replace("training.tif", "manual1.gif")
        )
        original_fov_path = os.path.join(
            augmented_fov_dir, image_name.replace(".tif", "_mask.gif")
        )

        image.save(original_image_path)
        mask.save(original_mask_path)
        fov.save(original_fov_path)

        for i in range(augmentations_per_image - 1):
            # Apply augmentations
            augmented_image, augmented_mask, augmented_fov = augment_image(
                image, mask, fov
            )

            # Save augmented image and mask
            augmented_image_path = os.path.join(
                augmented_image_dir, f"{os.path.splitext(image_name)[0].split("_")[0]}_aug_{i}_training.tif"
            )
            augmented_mask_path = os.path.join(
                augmented_mask_dir, f"{os.path.splitext(image_name)[0].split("_")[0]}_aug_{i}_manual1.gif"
            )
            augmented_fov_path = os.path.join(
                augmented_fov_dir, f"{os.path.splitext(image_name)[0].split("_")[0]}_aug_{i}_training_mask.gif"
            )

            augmented_image.save(augmented_image_path)
            augmented_mask.save(augmented_mask_path)
            augmented_fov.save(augmented_fov_path)

            # Stop if the desired number of augmented images is reached
            if len(os.listdir(augmented_image_dir)) >= desired_augmented_count:
                break

    print("Augmentation and saving completed.")
