import os
import random
from PIL import Image  # For image loading
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


def apply_fov(image, fov):
    if image.shape[0] == 3:
        fov = np.repeat(fov, 3, axis=0)

    return image * fov


def binarize_mask(mask):
    return np.uint8(mask > 0)


def apply_clahe(image):
    # Convert the image to a numpy array and transpose to HWC format
    # Convert to grayscale
    image = np.transpose(image, (1, 2, 0)).astype(np.float32)
    image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Ensure the image is in 8-bit format
    image_bw = (image_bw * 255).astype(np.uint8)
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=5.0)
    final_img = clahe.apply(image_bw)
    # Convert back to PIL Image
    return final_img


def load_img(image_path, size, standardize, channel):
    # If path is none return dummy image
    if image_path is None:
        arr = np.zeros((size[1], size[0]))
        arr = np.expand_dims(arr, axis=0)
        return arr

    # Step 3: Open the image file
    with Image.open(image_path) as img:
        img = img.resize(size)

        # Optional: Convert the image to a numpy array for numerical operations
        img_array = np.array(img)

    if channel in [0, 1, 2] and img_array.ndim == 3:
        img_array = img_array[:, :, channel]
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
        # transform=None,
        channel=None,
        clahe=False,
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
        # self.transform = transform
        self.channel = channel
        self.clahe = clahe

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
            load_img(image_path, self.size, self.standardize, self.channel),
            load_img(mask_path, self.size, self.standardize, self.channel),
            load_img(fov_path, self.size, self.standardize, self.channel),
        )
        mask, fov = binarize_mask(mask), binarize_mask(fov)

        # if self.transform:
        #     # image = np.transpose(image, (1, 2, 0))
        #     # mask = np.transpose(mask, (1, 2, 0))
        #     # fov = np.transpose(fov, (1, 2, 0))
        #     image, mask, fov = (
        #         torch.from_numpy(image),
        #         torch.from_numpy(mask),
        #         torch.from_numpy(fov),
        #     )
        #     # masks = [mask, fov]
        #     # image, mask, fov = self.transform(image, mask, fov)
        #     # # image = transformed["image"]
        #     # # mask, fov = transformed["masks"]

        #     # Random vertical flipping
        #     if random.random() > 0.5:
        #         image = TF.vflip(image)
        #         mask = TF.vflip(mask)
        #         fov = TF.vflip(fov)

        #     # Random rotation
        #     if random.random() > 0.5:
        #         angle = random.randint(0, 45)
        #         image = TF.rotate(image, angle)
        #         mask = TF.rotate(mask, angle)
        #         fov = TF.rotate(fov, angle)

        #     image, mask, fov = image.numpy(), mask.numpy(), fov.numpy()
        #     # image, mask, fov = (
        #     #     np.transpose(image, (2, 0, 1)),
        #     #     np.transpose(mask, (2, 0, 1)),
        #     #     np.transpose(fov, (2, 0, 1)),
        #     # )

        if self.clahe:
            image = apply_clahe(image)
            image = image / 255.0

        image, mask = apply_fov(image, fov), apply_fov(mask, fov)

        return image.astype(np.float32), mask.astype(np.float32), fov.astype(np.float32)
