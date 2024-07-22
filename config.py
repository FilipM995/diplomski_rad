import segmentation_models_pytorch as smp
import torch

from torch import nn

ROOT_DIR = "archive\\DRIVE"
TRAIN_ROOT_DIR = ROOT_DIR + "\\training"
TEST_ROOT_DIR = ROOT_DIR + "\\test"
TRAIN_SIZE = 15
VALIDATION_SIZE = 5

ENCODER = "resnet34"
ENCODER_WEIGHTS = "imagenet"
# DEVICE = "cpu"
MODE = smp.losses.constants.BINARY_MODE
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# create segmentation model with pretrained encoder
MODEL = smp.Unet(
    encoder_name=ENCODER,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights=ENCODER_WEIGHTS,  # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=1,  # model output channels (number of classes in your dataset)
)


EPOCHS = 25
SIZE = (512, 512)
BATCH_SIZE = 1
LEARNING_RATE = 0.001

LOSS = smp.losses.DiceLoss(mode=MODE, from_logits=True)
OPTIMIZER = torch.optim.Adam(MODEL.parameters(), lr=LEARNING_RATE)

DICE = True
AVERAGE = "micro"

AUC_METRIC = True

MIOU = True

IOU = False
ACCURACY = False
PRECISION = False
RECALL = False
F1 = False
