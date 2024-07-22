import torch
import torch.utils
import torch.utils.data
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import Dice
from torchmetrics.segmentation import MeanIoU
from sklearn.metrics import roc_auc_score
import time

from config import *


def train_one_epoch(
    epoch_index,
    tb_writer: SummaryWriter,
    training_loader,
):
    start_time = time.time()

    running_loss = 0.0
    last_loss = 0.0

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        image, mask, fov = data
        # image = image.float()
        image, mask, fov = (
            image.to(DEVICE),
            mask.to(DEVICE),
            fov.to(DEVICE),
        )  # Move data to the appropriate device

        # Zero your gradients for every batch!
        OPTIMIZER.zero_grad()

        # Make predictions for this batch
        outputs = MODEL(image)
        # print(torch.sum(outputs.eq(0)).item())
        # outputs = apply_fov(outputs, fov)
        # print(torch.sum(outputs.eq(0)).item())

        # Compute the loss and its gradients
        loss = LOSS(outputs, mask)
        loss.backward()

        # Adjust learning weights
        OPTIMIZER.step()

        # Gather data and report
        running_loss += loss.item()
        if (
            i % BATCH_SIZE == BATCH_SIZE - 1
        ):  # Adjust condition to dynamically match the batch size
            last_loss = (
                running_loss / BATCH_SIZE
            )  # Dynamically calculate loss per batch
            print("  batch {} loss: {}".format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar("Loss/train", last_loss, tb_x)
            running_loss = 0.0

    end_time = time.time()
    duration = end_time - start_time
    print(f"Finished in {duration:.2f} seconds")

    return last_loss


def save_one_img(
    epoch_number,
    loader: torch.utils.data.DataLoader,
    tb_writer: SummaryWriter,
):
    # Extract the image and mask from the specific_loader
    vimg_print, vmask_print, _ = next(iter(loader))
    vimg_print, vmask_print = vimg_print.to(DEVICE), vmask_print.to(DEVICE)
    voutputs_print = MODEL(vimg_print)

    # Convert the tensors to grid images for the specific image
    vimg_grid_print = torchvision.utils.make_grid(vimg_print)
    vmask_grid_print = torchvision.utils.make_grid(vmask_print)
    voutputs_grid_print = torchvision.utils.make_grid(
        torch.sigmoid(voutputs_print) > 0.5
    )  # Assuming binary classification

    # Log the images for the specific image
    tb_writer.add_image("Validation/Image", vimg_grid_print, epoch_number)
    tb_writer.add_image("Validation/Mask", vmask_grid_print, epoch_number)
    tb_writer.add_image("Validation/Output", voutputs_grid_print, epoch_number)


def calculate_metrics(preds: torch.Tensor, targets: torch.Tensor):
    metrics = {}
    if DICE:
        # Assure that the tnesors are integers
        preds_dice = preds.int()
        targets_dice = targets.int()
        dice = Dice(average=AVERAGE)
        metric = dice(preds_dice, targets_dice)
        metrics["dice"] = metric.numpy()

    if AUC_METRIC:
        # Assure that the tensors are 1 dimensional
        # Flatten the masks
        preds_auc = preds.flatten().cpu().numpy()
        targets_auc = targets.flatten().cpu().numpy()

        # Compute AUC-ROC
        auc = roc_auc_score(preds_auc, targets_auc)
        metrics["auc"] = auc

    if MIOU:
        preds_mean = preds.unsqueeze(0).int()
        targets_mean = targets.unsqueeze(0).int()
        miou = MeanIoU(num_classes=1)
        metric = miou(preds_mean, targets_mean)
        metrics["miou"] = metric.numpy()

    return metrics
