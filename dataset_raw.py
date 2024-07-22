import os
from PIL import Image  # For image loading
import numpy as np
import torch
from torch.utils.data import Dataset


def apply_fov(image, fov):
    if image.shape[0] == 3:
        fov = np.repeat(fov, 3, axis=0)

    return image * fov


def binarize_mask(mask):
    return np.uint8(mask > 0)


def load_img(image_path, size, standardize):
    # If path is none return dummy image
    if image_path is None:
        arr = np.zeros((size[0], size[1]))
        arr = np.expand_dims(arr, axis=0)
        return arr

    # Step 3: Open the image file
    with Image.open(image_path) as img:
        img = img.resize(size)

        # Optional: Convert the image to a numpy array for numerical operations
        img_array = np.array(img)
    # Normalize the image array
    img_array = img_array / 255.0

    if standardize:
        img_array = (img_array - np.mean(img_array)) / np.std(img_array)

    if len(img_array.shape) == 2:
        # For grayscale images, add a channel dimension
        img_array = np.expand_dims(img_array, axis=0)
    elif len(img_array.shape) == 3:
        img_array = img_array.transpose(2, 0, 1)

    return img_array


class DRIVEDataset(Dataset):

    def __init__(
        self,
        root_dir,
        size,
        identifiers=None,
        testing=False,
        standardize=False,
        transform=None,
    ):
        """
        Args:
            root_dir (str): Path to the root directory containing the dataset.
            mask_type (str, optional): Type of mask to use for training ('auto' or 'manual'). Defaults to 'auto'.
            transform (callable, optional): A function/transform to apply to the images. Defaults to None.
        """
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, "images")
        self.mask_dir = os.path.join(root_dir, "mask")
        self.mask_dir_manual = os.path.join(root_dir, "1st_manual")
        if identifiers is None:
            self.image_names = [
                f
                for f in os.listdir(self.image_dir)
                if os.path.isfile(os.path.join(self.image_dir, f))
            ]
        else:
            self.image_names = identifiers
        self.size = size
        self.testing = testing
        self.standardize = standardize
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_path = os.path.join(self.image_dir, image_name)

        if not self.testing:
            mask_path = os.path.join(
                self.mask_dir_manual,
                image_name.replace("_training.tif", "_manual1.gif"),
            )  # Assuming manual masks are in GIF format
        else:
            mask_path = None

        fov_path = os.path.join(
            self.mask_dir, image_name.replace(".tif", "_mask.gif")
        )  # Assuming masks are in GIF format

        image, mask, fov = (
            load_img(image_path, self.size, self.standardize),
            load_img(mask_path, self.size, self.standardize),
            load_img(fov_path, self.size, self.standardize),
        )
        mask, fov = binarize_mask(mask), binarize_mask(fov)

        if self.transform:
            image = np.transpose(image, (1, 2, 0))
            mask = np.transpose(mask, (1, 2, 0))
            fov = np.transpose(fov, (1, 2, 0))
            masks = [mask, fov]
            transformed = self.transform(image=image, masks=masks)
            image = transformed["image"]
            mask, fov = transformed["masks"]

            image, mask, fov = image.numpy(), mask.numpy(), fov.numpy()
            mask, fov = np.transpose(mask, (2, 0, 1)), np.transpose(fov, (2, 0, 1))

        image, mask = apply_fov(image, fov), apply_fov(mask, fov)

        return image.astype(np.float32), mask.astype(np.float32), fov.astype(np.float32)
