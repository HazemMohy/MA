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
    EnsureChannelFirstd,
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
import warnings
warnings.filterwarnings("ignore")  # remove some scikit-image warnings
##################################
#The pin_memory parameter in DataLoader is used to speed up data transfer between the host (CPU) and the GPU. When pin_memory is set to True, it allows the data loader to use pinned (page-locked) memory, which can make data transfer
#to the GPU faster. This is particularly useful when training models on a GPU.
#pin_memory = torch.cuda.is_available()
device = torch.device("cuda:0")
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
# Create dictionaries for the dataset

data_dicts = [
    {"image": img, "label": label}
    for img, label in zip(shuffled_paths, shuffled_labels_one_hot)
]

print(data_dicts[:5])
##################################
##################################
print("Create transforms")

# Define custom function for Lambda transform
def print_shape(x):
    print(f"Shape: {x.shape}")
    return x

# Define transforms
# I used the sames ones as in phase 1 (segmntation task), however, with one exception!
train_transforms = Compose([
    #The image_only parameter in the LoadImage transform in MONAI specifies whether to return only the image data or a dictionary containing additional metadata. When image_only=True,
    #the transform will load and return only the image itself, without any accompanying metadata. This is useful when you only need the image data for further processing or feeding into a neural network --> LoadImage(image_only=True)
    LoadImaged(keys=["image"]), 
    EnsureChannelFirstd(keys=["image"]), 
    #AddChannel(),
    NormalizeIntensityd(keys=["image"], nonzero=True),
    #ScaleIntensity(),
    #Resize((96, 96, 96)),
    SpatialPadd(keys=["image"], spatial_size=(320, 320, 320), mode='reflect'),
    Lambda(print_shape),
    #RandRotate90(),
    
    # ToTensord transform is typically used to convert both images and labels to PyTorch tensors, which is necessary for compatibility with
    #PyTorch models. Even if the labels contain only 1's and 0's, they should still be converted to tensors so that they can be used in computations with the model.
    ToTensord(keys=["image", "label"])
])

val_transforms = Compose([
    LoadImaged(keys=["image"]), 
    EnsureChannelFirstd(keys=["image"]), 
    NormalizeIntensityd(keys=["image"], nonzero=True),
    #ScaleIntensity(),
    #Resize((96, 96, 96)),
    SpatialPadd(keys=["image"], spatial_size=(320, 320, 320), mode='reflect'),
    Lambda(print_shape),
    #RandRotate90(),
    ToTensord(keys=["image", "label"])
])
##################################
print("Define dataset loaders")

# Prepare data: Split data into training and validation
# train_files = [{"image": img, "label": label} for img, label in zip(shuffled_paths[:7], shuffled_labels[:7])]
# val_files = [{"image": img, "label": label} for img, label in zip(shuffled_paths[-3:], shuffled_labels[-3:])]
train_files = data_dicts[:7]
val_files = data_dicts[-3:]

# # Define ImageDataset
# check_ds = ImageDataset(image_files=shuffled_paths, labels=shuffled_labels, transform=train_transforms)
# check_loader = DataLoader(check_ds, batch_size=1, num_workers=2, pin_memory=pin_memory) #batch_size = 1 ONLY in the testing_phase

# # Check first data loader output
# im, label = monai.utils.misc.first(check_loader) #Fetches the first batch from the data loader to check the output type and shape.
# print(type(im), im.shape, label, label.shape) #Prints the type and shape of the images and labels to verify correctness.

# Create a training data loader
# train_ds = ImageDataset(image_files=shuffled_paths[:7], labels=shuffled_labels[:7], transform=train_transforms) #Creates a dataset object using ImageDataset with the shuffled paths and labels, applying the train_transforms.
# train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=2, pin_memory=pin_memory) #Creates a data loader for the dataset. pin_memory=pin_memory enables pinned memory if a GPU is available.
train_ds = Dataset(data=train_files, transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=2)#, pin_memory=pin_memory)

# Create a validation data loader
# val_ds = ImageDataset(image_files=shuffled_paths[-3:], labels=shuffled_labels[-3:], transform=val_transforms)
# val_loader = DataLoader(val_ds, batch_size=1, num_workers=2, pin_memory=pin_memory)
val_ds = Dataset(data=val_files, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=2)#, pin_memory=pin_memory)


# Check first data loader output
im, label = monai.utils.misc.first(train_loader)
print(type(im), im.shape, label, label.shape)


# val_ds = Dataset(data=val_files, transform=val_transforms)
# val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=2)

# test_ds = Dataset(data=test_files, transform=test_transforms)
# test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=2)
##################################
# standard PyTorch program style
# Create DenseNet121, CrossEntropyLoss and Adam optimizer. THEN, start the Training & Evaluation 

# print("Create Model")
# model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=2).to(device)

# print("Create Loss")
# loss_function = torch.nn.CrossEntropyLoss()
# #loss_function = torch.nn.BCEWithLogitsLoss()  # also works with this data
# #loss_function = DiceCELoss(include_background=True, to_onehot_y=True, sigmoid=True) #MONAI-DiceCELoss #try softmax #CHANGE #it is NOT PURELY binary cross entropy loss

# print("Create Optimizer")
# learning_rate = 1e-4
# optimizer = torch.optim.Adam(model.parameters(), learning_rate)

# print("Training and Evaluation started!")
# val_interval = 1
# best_metric = -1
# best_metric_epoch = -1
# epoch_loss_values = []
# metric_values = []
# writer = SummaryWriter()
# max_epochs = 5 #just for the testing phase


# for epoch in range(max_epochs):
#     print("-" * 10)
#     print(f"epoch {epoch + 1}/{max_epochs}")
#     model.train()
#     epoch_loss = 0
#     step = 0

#     for batch_data in train_loader:
#         step += 1
#         inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = loss_function(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         epoch_loss += loss.item()
#         epoch_len = len(train_ds) // train_loader.batch_size
#         print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
#         writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)

#     epoch_loss /= step
#     epoch_loss_values.append(epoch_loss)
#     print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

#     if (epoch + 1) % val_interval == 0:
#         model.eval()

#         num_correct = 0.0
#         metric_count = 0
#         for val_data in val_loader:
#             val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
#             with torch.no_grad():
#                 val_outputs = model(val_images)
#                 value = torch.eq(val_outputs.argmax(dim=1), val_labels.argmax(dim=1))
#                 metric_count += len(value)
#                 num_correct += value.sum().item()

#         metric = num_correct / metric_count
#         metric_values.append(metric)

#         if metric > best_metric:
#             best_metric = metric
#             best_metric_epoch = epoch + 1
#             torch.save(model.state_dict(), "best_metric_model_classification3d_array.pth")
#             print("saved new best metric model")

#         print(f"Current epoch: {epoch+1} current accuracy: {metric:.4f} ")
#         print(f"Best accuracy: {best_metric:.4f} at epoch {best_metric_epoch}")
#         writer.add_scalar("val_accuracy", metric, epoch + 1)

# print(f"Training completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
# writer.close()
