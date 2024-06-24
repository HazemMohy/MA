print("START_Classification!")
##################################

import numpy as np
from monai.utils import first, set_determinism
from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    ConcatItemsd,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    ToTensord,
    AsChannelFirstD,
    RandFlipd,
    NormalizeIntensityd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    CropForegroundd,
    Rand2DElastic,
    RandAffined,
    SpatialPadd,
    Lambda,
    EnsureChannelFirst,
    RandRotate90,
    Resize,
    ScaleIntensity,
    )
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric #compute_meandice can NOT be found!!
from monai.losses import DiceLoss, TverskyLoss, DiceCELoss
from monai.inferers import sliding_window_inference
from monai.data import SmartCacheDataset, CacheDataset, DataLoader, Dataset, ImageDataset
from monai.config import print_config
from monai.apps import download_and_extract
from monai.optimizers import Novograd
import torch
from torch.nn import BCEWithLogitsLoss, BCELoss
torch.multiprocessing.set_sharing_strategy('file_system') #????????????
import sys
import matplotlib
import matplotlib.pyplot as plt
import tempfile
import shutil
import glob
import random
import csv
import pandas as pd
import nibabel as nib
from scipy.ndimage import zoom
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts
##############################################################################################################################################################################
#new imports

import logging
from torch.utils.tensorboard import SummaryWriter
import monai
##################################
import json
import os
##################################
pin_memory = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# print_config()
##################################

# Define the data directory path
data_dir = "/lustre/groups/iterm/Hazem/MA/data/4x"

# Get the first 5 .nii.gz files from bg directory
train_bg = sorted(glob.glob(os.path.join(data_dir, 'bg', "*.nii.gz")))[:5]

# Get the first 5 .nii.gz files from raw directory
train_raw = sorted(glob.glob(os.path.join(data_dir, 'raw', "*.nii.gz")))[:5]

# Check if the lists are correctly populated
if not train_bg:
    print("No .nii.gz files found in the bg directory.")
if not train_raw:
    print("No .nii.gz files found in the raw directory.")

# Combine the two lists into one variable
combined_paths = train_bg + train_raw

# Create binary labels: 0 for bg, 1 for raw
labels = np.array([0] * len(train_bg) + [1] * len(train_raw))

# Print the combined paths and their corresponding labels
print("Combined paths of bg and raw with labels:")
for path, label in zip(combined_paths, labels):
    print(f"{path} - Label: {label}")

# Pair paths and labels together and shuffle them
combined = list(zip(combined_paths, labels))
random.shuffle(combined)

# Separate the paths and labels back into individual lists after shuffling
shuffled_paths, shuffled_labels = zip(*combined)
# Print the shuffled paths and their corresponding labels
print("Shuffled paths of bg and raw with labels:")
for path, label in zip(shuffled_paths, shuffled_labels):
    print(f"{path} - Label: {label}")


# Convert labels to one-hot format
# Represent labels in one-hot format for binary classifier training
# BCEWithLogitsLoss requires target to have same shape as input
labels_one_hot = torch.nn.functional.one_hot(torch.as_tensor(labels)).float()

# Print the one-hot encoded labels
print("One-hot encoded labels:")
print(labels_one_hot)

