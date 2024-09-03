import os
import segmentation_models_pytorch as smp
import torch

# import albumentations as A
# from albumentations.pytorch import ToTensorV2
import torchvision.transforms.v2 as transforms


from torch import nn
from augment import augment_dataset

ROOT_DIR = "archive\\DRIVE"
TRAIN_ROOT_DIR = ROOT_DIR + "\\training"
TEST_ROOT_DIR = ROOT_DIR + "\\test"
VALIDATION_SIZE = 0.2

ENCODER = "vgg16"
ENCODER_WEIGHTS = "imagenet"
# DEVICE = "cpu"
MODE = smp.losses.constants.BINARY_MODE
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHANNEL = 3
CLAHE = True

if CHANNEL in [0, 1, 2] or CLAHE:
    IN_CHANNELS = 1
else:
    IN_CHANNELS = 3

MODEL_TYPE = "unet"

match MODEL_TYPE:
    case "unet":
        MODEL = smp.Unet(
            encoder_name=ENCODER,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=ENCODER_WEIGHTS,  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=IN_CHANNELS,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,  # model output channels (number of classes in your dataset)
        )
    case "fpn":
        MODEL = smp.FPN(
            encoder_name=ENCODER,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=ENCODER_WEIGHTS,  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=IN_CHANNELS,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,  # model output channels (number of classes in your dataset)
        )
    case "unet++":
        MODEL = smp.UnetPlusPlus(
            encoder_name=ENCODER,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=ENCODER_WEIGHTS,  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=IN_CHANNELS,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,  # model output channels (number of classes in your dataset)
        )
    case "pspnet":
        MODEL = smp.PSPNet(
            encoder_name=ENCODER,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=ENCODER_WEIGHTS,  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=IN_CHANNELS,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,  # model output channels (number of classes in your dataset)
        )
    case "deeplabv3":
        MODEL = smp.DeepLabV3(
            encoder_name=ENCODER,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=ENCODER_WEIGHTS,  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=IN_CHANNELS,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,  # model output channels (number of classes in your dataset)
        )
    case "deeplabv3+":
        MODEL = smp.DeepLabV3Plus(
            encoder_name=ENCODER,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=ENCODER_WEIGHTS,  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=IN_CHANNELS,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,  # model output channels (number of classes in your dataset)
        )
    case "linknet":
        MODEL = smp.Linknet(
            encoder_name=ENCODER,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=ENCODER_WEIGHTS,  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=IN_CHANNELS,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,  # model output channels (number of classes in your dataset)
        )
    case "manet":
        MODEL = smp.MAnet(
            encoder_name=ENCODER,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=ENCODER_WEIGHTS,  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=IN_CHANNELS,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,  # model output channels (number of classes in your dataset)
        )
    case "pan":
        MODEL = smp.PAN(
            encoder_name=ENCODER,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=ENCODER_WEIGHTS,  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=IN_CHANNELS,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,  # model output channels (number of classes in your dataset)
        )


EPOCHS = 25
SIZE = (512, 512)
BATCH_SIZE = 1
LEARNING_RATE = 0.001

LOSS_TYPE = "dice"

match LOSS_TYPE:
    case "bce":
        LOSS = nn.BCEWithLogitsLoss()
    case "dice":
        LOSS = smp.losses.DiceLoss(mode=MODE, from_logits=True)

OPTIMIZER_TYPE = "adam"
match OPTIMIZER_TYPE:
    case "adam":
        OPTIMIZER = torch.optim.Adam(MODEL.parameters(), lr=LEARNING_RATE)
    case "sgd":
        OPTIMIZER = torch.optim.SGD(MODEL.parameters(), lr=LEARNING_RATE)

DICE = True
AVERAGE = "micro"

AUC_METRIC = True

MIOU = True

IOU = False
ACCURACY = False
PRECISION = False
RECALL = True
F1 = True
MATTHEWS = True

METRICS = [
    metric
    for metric, flag in [
        ("auc", AUC_METRIC),
        ("miou", MIOU),
        ("iou", IOU),
        ("accuracy", ACCURACY),
        ("precision", PRECISION),
        ("recall", RECALL),
        ("f1", F1),
        ("matthews", MATTHEWS),
        ("dice", DICE),
    ]
    if flag
]


AUGMENTATION = False

if AUGMENTATION:
    augment_dataset(
        original_image_dir=os.path.join(TRAIN_ROOT_DIR, "images"),
        original_mask_dir=os.path.join(TRAIN_ROOT_DIR, "1st_manual"),
        original_fov_dir=os.path.join(TRAIN_ROOT_DIR, "mask"),
        augmented_image_dir="augment/images",
        augmented_mask_dir="augment/1st_manual",
        augmented_fov_dir="augment/mask",
        desired_augmented_count=50,
    )

    TRAIN_ROOT_DIR = "augment"

# if AUGMENTATION:
#     TRAIN_TRANSFORMS = transforms.Compose(
#         [
#             transforms.RandomHorizontalFlip(p=1),
#             transforms.RandomRotation(degrees=90),
#         ]
#     )

#     # TEST_TRANSFORMS = transforms.Compose(
#     #     [
#     #         ToTensorV2(),
#     #     ]
#     # )
#     TEST_TRANSFORMS = None


# else:
#     TRAIN_TRANSFORMS = None
#     TEST_TRANSFORMS = None
