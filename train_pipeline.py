import os
import shutil
from sklearn.model_selection import train_test_split
import torch
import torch.utils
from dataset_raw import DRIVEDataset
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Subset
from train_utils import train_one_epoch, save_one_img, calculate_metrics
from config import *
import os
import glob

print("Loading data...")

data_dir = TRAIN_ROOT_DIR + "\\images"

identifiers = [
    f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))
]

train_identifiers, val_identifiers = train_test_split(
    identifiers, test_size=VALIDATION_SIZE, random_state=42
)

train_data = DRIVEDataset(
    root_dir=TRAIN_ROOT_DIR,
    size=SIZE,
    identifiers=train_identifiers,
    channel=CHANNEL,
    clahe=CLAHE,
)
valid_data = DRIVEDataset(
    root_dir=TRAIN_ROOT_DIR,
    size=SIZE,
    identifiers=val_identifiers,
    channel=CHANNEL,
    clahe=CLAHE,
)
test_data = DRIVEDataset(
    root_dir=TEST_ROOT_DIR,
    size=SIZE,
    testing=True,
    channel=CHANNEL,
    clahe=CLAHE,
)


training_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
validation_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

# Assuming `dataset` is your validation dataset and `specific_image_index` is the index of the image you want to print
specific_image_index = (
    0  # Example index, replace with the actual index of your specific image
)
specific_dataset = Subset(valid_data, [specific_image_index])
specific_loader = DataLoader(specific_dataset, batch_size=1)

print("Data loaded successfully")

print("Clearing directories...")


def delete_old_files(directory, keep_last=10):
    files = glob.glob(os.path.join(directory, "*"))
    files.sort(key=os.path.getmtime, reverse=True)
    for file in files[keep_last:]:
        try:
            if os.path.isfile(file):
                os.remove(file)
            elif os.path.isdir(file):
                shutil.rmtree(file)
        except Exception as e:
            print(f"Error deleting file {file}: {e}")


# Directories to clean
directories = ["models", "runs"]

for directory in directories:
    delete_old_files(directory)

print(f"Using {DEVICE} device")

# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
writer = SummaryWriter(
    "runs/trainer_{}_{}_a={}_lr={}_bs={}_clahe={}_{}".format(
        MODEL_TYPE, ENCODER, AUGMENTATION, LEARNING_RATE, BATCH_SIZE, CLAHE, timestamp
    )
)
print("Model:")
print(MODEL)

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
        model_path = "models\\model_{}_{}_{}".format(
            MODEL_TYPE, timestamp, epoch_number
        )
        torch.save(MODEL.state_dict(), model_path)

    epoch_number += 1

writer.close()


def evaluate_model(validation_loader, model, device):
    model.eval()  # Set the model to evaluation mode
    metrics_sum = {metric: 0 for metric in METRICS}
    num_batches = 0

    with torch.no_grad():  # Disable gradient calculation
        for img, mask, fov in validation_loader:
            img, mask = img.to(device), mask.to(device)
            outputs = model(img).cpu()
            outputs = torch.sigmoid(outputs) > 0.5
            mask = mask.cpu()

            batch_metrics = calculate_metrics(mask.squeeze(), outputs.squeeze())
            for key in metrics_sum:
                metrics_sum[key] += batch_metrics[key]
            num_batches += 1

    # Calculate average metrics
    avg_metrics = {key: value / num_batches for key, value in metrics_sum.items()}
    return avg_metrics


print(evaluate_model(validation_loader, MODEL, DEVICE))
