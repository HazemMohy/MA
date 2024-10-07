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
import torch.nn as nn
import torch.nn.functional as F
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
#SLURM_JOB_ID
slurm_job_id = os.environ.get('SLURM_JOB_ID', 'default_job_id')

#"Runs" directory
runs_dir = "/lustre/groups/iterm/Hazem/MA/Runs"
os.makedirs(runs_dir, exist_ok=True)

#specific "Run" directory for a specific run
#run_folder_name = f"run_{slurm_job_id}__{loss_function_name}_{chosen_scheduler_name}"
run_folder_name = f"run_{slurm_job_id}__Phase_2"
run_dir = os.path.join(runs_dir, run_folder_name)
os.makedirs(run_dir, exist_ok=True)


# # Define the path to save the best model
# save_path = os.path.join(run_dir, f"best_metric_model_classification3d_array_{slurm_job_id}_{max_epochs}_{learning_rate}.pth")
##################################



# Define the data directory path
data_dir = "/lustre/groups/iterm/Hazem/MA/data/4x"

# Get the first 5 .nii.gz files from bg directory, NOPE --> Get ALL now!
train_bg = sorted(glob.glob(os.path.join(data_dir, 'bg', "*.nii.gz")))

# Get the first 5 .nii.gz files from raw directory, NOPE --> Get ALL now!
train_raw = sorted(glob.glob(os.path.join(data_dir, 'raw', "*.nii.gz")))

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
random.seed(42) # Ensure reproducibility
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
#One-hot encoding is a method of converting categorical labels into a binary vector format where only one element is "hot" (i.e., set to 1) and all other elements are "cold" (i.e., set to 0). This is
#particularly useful in machine learning tasks because many algorithms, including neural networks, require numerical input and cannot work directly with categorical labels.
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

print("Data before shuffling:")
for d in data_dicts:
    print(d)

# Shuffle the data to ensure randomness
random.seed(42)
random.shuffle(data_dicts)

print("Data after shuffling:")
for d in data_dicts:
    print(d)
##################################
# DATA SPLIT with the requirement the the testing dataset has the same patches as the one in phase 1!!

# Define the test set with appropriate labels
test_files = [
    {"image": "/lustre/groups/iterm/Hazem/MA/data/4x/bg/patchvolume_61.nii.gz", "label": torch.tensor([1., 0.])},
    {"image": "/lustre/groups/iterm/Hazem/MA/data/4x/raw/patchvolume_61.nii.gz", "label": torch.tensor([0., 1.])},
    {"image": "/lustre/groups/iterm/Hazem/MA/data/4x/bg/patchvolume_26.nii.gz", "label": torch.tensor([1., 0.])},
    {"image": "/lustre/groups/iterm/Hazem/MA/data/4x/raw/patchvolume_26.nii.gz", "label": torch.tensor([0., 1.])},
    {"image": "/lustre/groups/iterm/Hazem/MA/data/4x/bg/patchvolume_40.nii.gz", "label": torch.tensor([1., 0.])},
    {"image": "/lustre/groups/iterm/Hazem/MA/data/4x/raw/patchvolume_40.nii.gz", "label": torch.tensor([0., 1.])},
    {"image": "/lustre/groups/iterm/Hazem/MA/data/4x/bg/patchvolume_256.nii.gz", "label": torch.tensor([1., 0.])},
    {"image": "/lustre/groups/iterm/Hazem/MA/data/4x/raw/patchvolume_256.nii.gz", "label": torch.tensor([0., 1.])},
]

# Remove test set files from data_dicts
test_files_set = set(item["image"] for item in test_files)
data_dicts = [item for item in data_dicts if item["image"] not in test_files_set]

# Calculate the number of patches for training and validation
num_total = len(data_dicts)
num_val = max(1, round(num_total * 0.2))
num_train = num_total - num_val

# Split the remaining patches into training and validation
train_files = data_dicts[:num_train]
val_files = data_dicts[num_train:]

print(f"Total patches: {num_total}")
print(f"Training patches: {num_train}")
print(f"Validation patches: {num_val}")
print(f"Testing patches: {len(test_files)}")

print("Train files:")
for file in train_files:
    print(file)

print("Validation files:")
for file in val_files:
    print(file)

print("Test files:")
for file in test_files:
    print(file)


# from here till the bottom
##################################
print("Create transforms")

# Define custom function for Lambda transform
# def print_shape(x):
#     print(f"Shape: {x.shape}")
#     return x

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
    #Lambda(print_shape),
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
    #Lambda(print_shape),
    #RandRotate90(),
    ToTensord(keys=["image", "label"])
])

test_transforms = Compose([
    LoadImaged(keys=["image"]), 
    EnsureChannelFirstd(keys=["image"]), 
    NormalizeIntensityd(keys=["image"], nonzero=True),
    #ScaleIntensity(),
    #Resize((96, 96, 96)),
    SpatialPadd(keys=["image"], spatial_size=(320, 320, 320), mode='reflect'),
    #Lambda(print_shape),
    #RandRotate90(),
    ToTensord(keys=["image", "label"])
])
##################################
print("Define dataset loaders")

# Prepare data: Split data into training and validation
# train_files = [{"image": img, "label": label} for img, label in zip(shuffled_paths[:7], shuffled_labels[:7])]
# val_files = [{"image": img, "label": label} for img, label in zip(shuffled_paths[-3:], shuffled_labels[-3:])]
# train_files = data_dicts[:7]
# val_files = data_dicts[-3:]

# # Define ImageDataset
# check_ds = ImageDataset(image_files=shuffled_paths, labels=shuffled_labels, transform=train_transforms)
# check_loader = DataLoader(check_ds, batch_size=1, num_workers=2, pin_memory=pin_memory) #batch_size = 1 ONLY in the testing_phase

# # Check first data loader output
# im, label = monai.utils.misc.first(check_loader) #Fetches the first batch from the data loader to check the output type and shape.
# print(type(im), im.shape, label, label.shape) #Prints the type and shape of the images and labels to verify correctness.

# Create a training data loader
# train_ds = ImageDataset(image_files=shuffled_paths[:7], labels=shuffled_labels[:7], transform=train_transforms) #Creates a dataset object using ImageDataset with the shuffled paths and labels, applying the train_transforms.
# train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=2, pin_memory=pin_memory) #Creates a data loader for the dataset. pin_memory=pin_memory enables pinned memory if a GPU is available.
train_ds = Dataset(data=train_files, transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4)#, pin_memory=pin_memory) #test --> batch_size = 2

# Create a validation data loader
# val_ds = ImageDataset(image_files=shuffled_paths[-3:], labels=shuffled_labels[-3:], transform=val_transforms)
# val_loader = DataLoader(val_ds, batch_size=4, num_workers=2, pin_memory=pin_memory)
val_ds = Dataset(data=val_files, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=2, num_workers=2)#, pin_memory=pin_memory)

# Create a testing data loader
test_ds = Dataset(data=test_files, transform=test_transforms)
test_loader = DataLoader(test_ds, batch_size=2, num_workers=2)


# # Check first data loader output
# im, label = monai.utils.misc.first(train_loader)
# print(type(im), im.shape, label, label.shape)


# val_ds = Dataset(data=val_files, transform=val_transforms)
# val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=2)

# test_ds = Dataset(data=test_files, transform=test_transforms)
# test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=2)
##################################
#standard PyTorch program style
#Create U-Net_for_Classification, CrossEntropyLoss and Adam optimizer. THEN, start the Training & Evaluation 
# Custom UNet for Classification

print("Create U-Net for Classification")
#Best Option for Adapting U-Net for Binary Classification: having the MONAI-U-Net as it is (remains UNCHANGED) THEN using/ADDING a global average pooling followed by a fully connected layer, which are standard techniques in CNN-based classification tasks.
#NOT replacing the last layer of MONAI's U-Net and implementing instead a fully connected layer. That is an option as well BUT NOT the most optimal one! WHY adding is better than replacement?
#   1) It leverages the feature extraction capabilities of the original U-Net and adapts the output for classification without altering the core architecture of the U-Net.
#   2) Modularity: This approach maintains the integrity of the original U-Net model, which makes it easier to update or modify the base model without affecting the classification layers.
#   3) Simplicity: Adding layers for pooling and classification is simpler and less error-prone than modifying the original U-Net structure.
#   4) Flexibility: This method allows you to change the classification layers independently of the base model, making it easier to experiment with different pooling and classification strategies.
#   5) Separation of Concerns: By adding layers, you separate the feature extraction (handled by U-Net) from the classification (handled by the added layers), which is a cleaner design.
#This method efficiently reduces the spatial dimensions and produces a single output for classification
# ZUSAMMENFASSEND:
# - The custom UNet model for classification first extracts features using the UNet architecture.
# - The extracted features are globally pooled and then passed through a fully connected layer to obtain class scores. 
# - This approach adapts the segmentation capabilities of UNet for a classification task by aggregating the spatial features and mapping them to class probabilities.
class UNetForClassification(nn.Module):
    def __init__(self):
        super(UNetForClassification, self).__init__()
        self.unet = UNet(
            spatial_dims=3,
            in_channels=1,

            # out_channels --> This defines the depth of the feature maps output by the UNet. This value can be adjusted based on the complexity of the data and the desired model capacity.
            # The value 32 here is not related to the number of classes but rather to the richness of the feature representation extracted by the UNet. A higher value allows the network to learn more complex features but increases the computational load.
            out_channels=32,  # Adjust this based on the UNet implementation
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2)
        )
        # Global average pooling
        # This layer converts the feature map into a fixed-size vector by averaging the spatial dimensions.
        # Adding a global pooling layer before the final fully connected layer can aggregate the features from the spatial dimensions into a single vector suitable for classification.
        # This layer reduces the spatial dimensions of the feature map output from the UNet to 1x1x1 by taking the average value of each channel. This results in a tensor of shape (batch_size, out_channels, 1, 1, 1),
        # effectively converting the 3D feature map into a vector of size equal to out_channels (32 in this case).
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)  
        
        # Fully connected layer for classification
        # The output from the global pooling layer can be passed through a fully connected layer to get the class scores.
        # This layer maps the feature vector to the desired number of classes (2 in this case).
        # This layer maps the 32-dimensional feature vector to a single output.
        # nn.Linear(32, 1): This means we have 32 input features (from the 32 channels output by the global average pooling) and 1 output feature. The single output feature is suitable for binary classification.
        # The fully connected layer takes the 32 features from the global average pooling layer and maps them to a single output neuron. This single neuron output, after the sigmoid activation, gives the probability of
        # the input belonging to the positive class (foreground).
        self.fc = nn.Linear(32, 1) # 1 NOT 2 (BINARY classification) #Since you're using a sigmoid activation for binary classification, your output layer should have a single neuron.

    def forward(self, x):
        x = self.unet(x) # Passes the input through the UNet to extract features.
        x = self.global_avg_pool(x) # Applies global average pooling to reduce the spatial dimensions.
        x = torch.flatten(x, 1) # Flattens the tensor into a shape suitable for the fully connected layer.
        x = self.fc(x) # Passes the flattened features through the fully connected layer to produce a single output.
        #x = F.softmax(x, dim=1) #Ensure the output of your fc layer uses a softmax activation for a proper probability distribution over classes.
        x = torch.sigmoid(x) # Use sigmoid activation for binary classification. It ensures that the output is a probability value between 0 and 1, which is appropriate for binary classification. 
        return x

#################################
print("Create Model")
#model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=2).to(device)
model = UNetForClassification().to(device)

print("Create Loss")
#loss_function = torch.nn.CrossEntropyLoss() #OLD
loss_function = torch.nn.BCEWithLogitsLoss()  # Use BCEWithLogitsLoss for binary classification with a single output neuron. This loss function is suitable for binary classification tasks and expects RAW logits (NOT hot-encoded) as input.
#loss_function = DiceCELoss(include_background=True, to_onehot_y=True, sigmoid=True) #MONAI-DiceCELoss #try softmax #CHANGE #it is NOT PURELY binary cross entropy loss

print("Create Optimizer")
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), learning_rate)

print("Training and Evaluation started!")
val_interval = 20
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []
writer = SummaryWriter()
max_epochs = 500 #test --> 500 or 1000 (I want the real one to be 2000 but I get always a CUDA_OutOfMemory_Error)

# Define the path to save the best model
save_path = os.path.join(run_dir, f"best_metric_model_classification3d_array_{slurm_job_id}_{max_epochs}_{learning_rate}.pth")


for epoch in range(max_epochs):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0

    for batch_data in train_loader:
        step += 1
        #inputs, labels = batch_data[0].to(device), batch_data[1].to(device) #!!!!!!!! --> inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
        inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
        #labels = labels.argmax(dim=1).float().unsqueeze(1)  #OLD 
        #Ensure that the output handling in the training and evaluation loops matches the expected format for the loss function and accuracy computation.
        labels = labels[:, 1].unsqueeze(1)  # Use only the positive class for binary classification # BCEWithLogitsLoss expects single value labels # Convert one-hot to single value
        optimizer.zero_grad()
        outputs = model(inputs)
        #loss = loss_function(outputs, labels) #!!!!!!!! --> loss = loss_function(outputs, labels.argmax(dim=1))
        #CrossEntropyLoss: This loss function expects the target labels to be in class indices (not one-hot encoded). If your labels are one-hot encoded, you need to convert them back to class indices using labels.argmax(dim=1).
        #HERE: BCELoss or BCEWithLogitsLoss (you can use labels DIRECTLY): These loss functions can work directly with one-hot encoded labels or multi-label binary classification problems. If your network outputs match the shape and form of your one-hot encoded labels, you can use them directly.
        #loss = loss_function(outputs, labels.argmax(dim=1)) #OLD
        loss = loss_function(outputs, labels)  # Use BCEWithLogitsLoss which expects single value labels
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_len = len(train_ds) // train_loader.batch_size
        print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
        writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)

    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    if (epoch + 1) % val_interval == 0:
        model.eval()

        num_correct = 0.0
        metric_count = 0
        for val_data in val_loader:
            #val_images, val_labels = val_data[0].to(device), val_data[1].to(device) #!!!!!!!!! ---> val_images, val_labels = val_data["image"].to(device), val_data["label"].to(device)
            val_images, val_labels = val_data["image"].to(device), val_data["label"].to(device)
            #NIX #OLD
            val_labels = val_labels[:, 1].unsqueeze(1)  # Use only the positive class for binary classification
            #val_labels = val_labels.argmax(dim=1).float().unsqueeze(1)  # Convert one-hot to single value
            with torch.no_grad():
                val_outputs = model(val_images)
                #value = torch.eq(val_outputs.argmax(dim=1), val_labels.argmax(dim=1)) #OLD
                value = torch.eq((val_outputs > 0.5).float(), val_labels)  # Compare with threshold 0.5
                metric_count += len(value)
                num_correct += value.sum().item()

        metric = num_correct / metric_count
        metric_values.append(metric)

        if metric > best_metric:
            best_metric = metric
            best_metric_epoch = epoch + 1
            #torch.save(model.state_dict(), "best_metric_model_classification3d_array.pth")
            torch.save(model.state_dict(), save_path)
            print("saved new best metric model")

        print(f"Current epoch: {epoch+1} current accuracy: {metric:.4f} ")
        print(f"Best accuracy: {best_metric:.4f} at epoch {best_metric_epoch}")
        writer.add_scalar("val_accuracy", metric, epoch + 1)

print(f"Training completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
writer.close()
##################################
print("-" * 40)
print("Testing started!")

# test data loader already created

# Load the best model
#model.load_state_dict(torch.load("best_metric_model_classification3d_array.pth"))
model.load_state_dict(torch.load(save_path))
model.eval()

num_correct = 0.0
metric_count = 0

with torch.no_grad():
    for test_data in test_loader:
        test_images, test_labels = test_data["image"].to(device), test_data["label"].to(device)
        test_labels = test_labels[:, 1].unsqueeze(1)  # Use only the positive class for binary classification
        test_outputs = model(test_images)
        value = torch.eq((test_outputs > 0.5).float(), test_labels)  # Compare with threshold 0.5
        metric_count += len(value)
        num_correct += value.sum().item()

test_accuracy = num_correct / metric_count

print(f"Test accuracy: {test_accuracy:.4f}")

##################################
## the Runs folder - all in one   

# Save output and error at Runs
slurm_output_file = f"/lustre/groups/iterm/Hazem/MA/HPC/slurm_outputs/3D_Seg_{slurm_job_id}_output.txt"
slurm_error_file = f"/lustre/groups/iterm/Hazem/MA/HPC/slurm_outputs/3D_Seg_{slurm_job_id}_error.txt"
run_slurm_output_file = os.path.join(run_dir, f"3D_Classification_{slurm_job_id}_0_output_{best_metric:.4f}_{best_metric_epoch}_{test_accuracy:.4f}.txt")
run_slurm_error_file = os.path.join(run_dir, f"3D_Classification_{slurm_job_id}_0_error_{best_metric:.4f}_{best_metric_epoch}_{test_accuracy:.4f}.txt")

shutil.copy(slurm_output_file, run_slurm_output_file)
shutil.copy(slurm_error_file, run_slurm_error_file)
print(f"Slurm output file copied to {run_slurm_output_file}")
print(f"Slurm error file copied to {run_slurm_error_file}")

##################################
#final print
print("-" * 40)
print("ALL DONE!") 