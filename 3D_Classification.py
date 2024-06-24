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
    LoadImage,
    AddChannel,
    NormalizeIntensity,
    ScaleIntensity,
    EnsureChannelFirst,
    Resize,
    RandRotate90,
    ToTensor,
    SpatialPad,
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
#The pin_memory parameter in DataLoader is used to speed up data transfer between the host (CPU) and the GPU. When pin_memory is set to True, it allows the data loader to use pinned (page-locked) memory, which can make data transfer
#to the GPU faster. This is particularly useful when training models on a GPU.
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
# pairs each path with its corresponding label, creating tuples. AND "list" converts the zipped object into a list of tuples.
combined = list(zip(combined_paths, labels))
random.shuffle(combined) #shuffles the list of tuples, maintaining the correct path-label pairs.

# Separate the paths and labels back into individual lists after shuffling
#unzips the list of tuples back into two separate tuples: one for paths and one for labels.
#WHY? Most data processing and machine learning libraries expect features and labels to be provided separately. Separating them into individual lists aligns with this convention.
#and because as well: Functions and methods used in data loaders, training loops, and other processing steps often expect separate lists of features and labels.
shuffled_paths, shuffled_labels = zip(*combined)  #shuffled_paths, shuffled_labels = zip(*combined) assigns these tuples to separate variables.

# Print the shuffled paths and their corresponding labels
print("Shuffled paths of bg and raw with labels:")
for path, label in zip(shuffled_paths, shuffled_labels):
    print(f"{path} - Label: {label}")


# Convert SHUFFLED labels (NOT the original "labels") to one-hot format
# Represent labels in one-hot format for binary classifier training
# BCEWithLogitsLoss requires target to have same shape as input
shuffled_labels_one_hot = torch.nn.functional.one_hot(torch.as_tensor(shuffled_labels)).float()

# Print the one-hot encoded labels
print("One-hot encoded SHUFFLED labels:")
print(shuffled_labels_one_hot)
##################################
print("Create transforms")

# Define custom function for Lambda transform
def print_shape(x):
    print(f"Shape: {x.shape}")
    return x

# Define transforms
# I used the sames ones as in phase 1 (segmntation task)
train_transforms = Compose([
    #The image_only parameter in the LoadImage transform in MONAI specifies whether to return only the image data or a dictionary containing additional metadata. When image_only=True,
    #the transform will load and return only the image itself, without any accompanying metadata. This is useful when you only need the image data for further processing or feeding into a neural network.
    LoadImage(image_only=True), 
    
    AddChannel(),
    NormalizeIntensity(nonzero=True),
    #ScaleIntensity(),
    #EnsureChannelFirst(),
    #Resize((96, 96, 96)),
    SpatialPad(spatial_size=(320, 320, 320), mode='reflect'),
    Lambda(print_shape),
    #RandRotate90(),
    ToTensor()
])

val_transforms = Compose([
    LoadImage(image_only=True), 
    AddChannel(),
    NormalizeIntensity(nonzero=True),
    #ScaleIntensity(),
    #EnsureChannelFirst(),
    #Resize((96, 96, 96)),
    SpatialPad(spatial_size=(320, 320, 320), mode='reflect'),
    Lambda(print_shape),
    #RandRotate90(),
    ToTensor()
])
##################################
print("Define dataset loaders")

# Define ImageDataset
check_ds = ImageDataset(image_files=shuffled_paths, labels=shuffled_labels, transform=train_transforms)
check_loader = DataLoader(check_ds, batch_size=1, num_workers=2, pin_memory=pin_memory) #batch_size = 1 ONLY in the testing_phase

# Check first data loader output
im, label = monai.utils.misc.first(check_loader) #Fetches the first batch from the data loader to check the output type and shape.
print(type(im), im.shape, label, label.shape) #Prints the type and shape of the images and labels to verify correctness.

# Create a training data loader
train_ds = ImageDataset(image_files=shuffled_paths[:7], labels=shuffled_labels[:7], transform=train_transforms) #Creates a dataset object using ImageDataset with the shuffled paths and labels, applying the train_transforms.
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=2, pin_memory=pin_memory) #Creates a data loader for the dataset. pin_memory=pin_memory enables pinned memory if a GPU is available.

# Create a validation data loader
val_ds = ImageDataset(image_files=shuffled_paths[-3:], labels=shuffled_labels[-3:], transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=2, pin_memory=pin_memory)
##################################
