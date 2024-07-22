import torch
import numpy as np
import torch.utils
from DataExplore.dataset_raw import DRIVEDataset, TransformedDataset
import torchvision
import segmentation_models_pytorch as smp
import pytorch_lightning as pl
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Subset, random_split
from train_utils import train_one_epoch, save_one_img, calculate_metrics
from config import *
from torchvision import transforms


train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])
 
test_transforms = transforms.Compose([
    transforms.ToTensor(),
])

print("Loading data...")

train_data = DRIVEDataset(root_dir=TRAIN_ROOT_DIR, size=SIZE)
test_data = DRIVEDataset(root_dir=TEST_ROOT_DIR, size=SIZE, testing=True)

train_dataset, validation_dataset = random_split(
    train_data, [TRAIN_SIZE, VALIDATION_SIZE]
)

train_dataset=TransformedDataset(train_dataset, train_transforms)

training_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

# Assuming `dataset` is your validation dataset and `specific_image_index` is the index of the image you want to print
specific_image_index = (
    0  # Example index, replace with the actual index of your specific image
)
specific_dataset = Subset(validation_dataset, [specific_image_index])
specific_loader = DataLoader(specific_dataset, batch_size=1)

print("Data loaded successfully")

print(f"Using {DEVICE} device")

# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
writer = SummaryWriter("runs/fashion_trainer_{}".format(timestamp))

MODEL.to(DEVICE)  # Move MODEL to the appropriate DEVICE


epoch_number = 0

best_vloss = 1_000_000.0

for epoch in range(EPOCHS):
    print("EPOCH {}:".format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    MODEL.train(True)
    avg_loss = train_one_epoch(
        epoch_number,
        writer,
        training_loader,
    )

    running_vloss = 0.0
    # Set the MODEL to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    MODEL.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            vimg, vmask, _ = vdata
            vimg, vmask = vimg.to(DEVICE), vmask.to(
                DEVICE
            )  # Move data to the appropriate DEVICE

            voutputs = MODEL(vimg)
            vloss = LOSS(voutputs, vmask)
            running_vloss += vloss
        voutputs = (torch.sigmoid(voutputs) > 0.5).float().cpu()
        vmask = vmask.cpu()
        vmetrics = calculate_metrics(vmask.squeeze(), voutputs.squeeze())

    for metric_name, metric_value in vmetrics.items():
        writer.add_scalar(f"Validation/{metric_name}", metric_value, epoch_number + 1)

    avg_vloss = running_vloss / (i + 1)
    print("LOSS train {} valid {}".format(avg_loss, avg_vloss))

    save_one_img(epoch_number, specific_loader, writer)

    # Log the running LOSS averaged per batch
    # for both training and validation
    writer.add_scalars(
        "Training vs. Validation LOSS",
        {"Training": avg_loss, "Validation": avg_vloss},
        epoch_number + 1,
    )
    writer.flush()

    # Track best performance, and save the MODEL's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = "models\\model_{}_{}".format(timestamp, epoch_number)
        torch.save(MODEL.state_dict(), model_path)

    epoch_number += 1

writer.close()

img, mask, fov = next(iter(validation_loader))
img, mask, fov = img.to(DEVICE), mask.to(DEVICE), fov.to(DEVICE)
outputs = MODEL(img).cpu()
outputs = torch.sigmoid(outputs) > 0.5
mask = mask.cpu()


print(calculate_metrics(mask.squeeze(), outputs.squeeze()))
