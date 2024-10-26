print("START!")
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
    )
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric #compute_meandice can NOT be found!!
from monai.losses import DiceLoss, TverskyLoss, DiceCELoss
from monai.inferers import sliding_window_inference
from monai.data import SmartCacheDataset, CacheDataset, DataLoader, Dataset
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
import json
import os
##################################

import warnings
warnings.filterwarnings("ignore")  # remove some scikit-image warnings
#print_config()

##################################
import time
start_time = time.time()
##################################
#SLURM_JOB_ID
slurm_job_id = os.environ.get('SLURM_JOB_ID', 'default_job_id')

##################################
# Load hyperparameters from the JSON file
with open('hyperparameters.json', 'r') as f:
    hyperparameters = json.load(f)


# Extract hyperparameters
dataset_choice = hyperparameters["dataset_choice"]
learning_rate = hyperparameters["learning_rate"]
max_epochs = hyperparameters["max_epochs"]

# Conditional batch size and val_interval based on max_epochs
if max_epochs in [10, 50, 100]:
    batch_size = 4 #immer 8
    val_interval = 5
elif max_epochs in [1000, 4000, 8000]:
    batch_size = 4
    val_interval = 20
else:
    batch_size = 4
    val_interval = 20

loss_function_name = "DiceCELoss"
##################################
# Extract scheduler parameters if present
scheduler_params = hyperparameters.get("scheduler", None)

# Extract the chosen scheduler from the hyperparameters
chosen_scheduler_name = hyperparameters.get("chosen_scheduler", None)


# Choose which scheduler to use based on the chosen_scheduler parameter
chosen_scheduler = scheduler_params.get(chosen_scheduler_name) if scheduler_params and chosen_scheduler_name else None

#extract the scheduler type for logging
scheduler_info = f"Scheduler: {chosen_scheduler['type']}" if chosen_scheduler else "No Scheduler"

#Hyperparameters Confirmation
print(f"Scheduler: {scheduler_info}")
print(f"Loss Function: {loss_function_name}")
print(f"Dataset Choice: {dataset_choice}")
print(f"Learning Rate: {learning_rate}")
print(f"Max Epochs: {max_epochs}")
print(f"Batch Size: {batch_size}")
print(f"Validation Interval: {val_interval}")
#print(f"Number of Workers: {num_workers}")
##################################
##############################################################################################################################################################################
#Defining the directories
testing_dataset_dir = "/lustre/groups/iterm/Hazem/MA/Testing_Dataset"
os.makedirs(testing_dataset_dir, exist_ok=True)

my_models_dir = "/lustre/groups/iterm/Hazem/MA/models"
os.makedirs(my_models_dir, exist_ok=True)

my_plots_dir = "/lustre/groups/iterm/Hazem/MA/plots"
os.makedirs(my_plots_dir, exist_ok=True)

tracking_dir = "/lustre/groups/iterm/Hazem/MA/Tracking"
os.makedirs(tracking_dir, exist_ok=True)

runs_dir = "/lustre/groups/iterm/Hazem/MA/Runs"
os.makedirs(runs_dir, exist_ok=True)
##################################

loss_function_name = "DiceCELoss"
run_folder_name = f"run_{slurm_job_id}__{loss_function_name}_{chosen_scheduler_name}_ECXLUDE_ENCODER"
run_dir = os.path.join(runs_dir, run_folder_name)
os.makedirs(run_dir, exist_ok=True)
##################################

# Define directory to save NIfTI files
output_nifti_dir = os.path.join(run_dir, "Model_outputs_nifti")
os.makedirs(output_nifti_dir, exist_ok=True)

##################################
#mixed loss_function

class MixedLoss(torch.nn.Module):
    def __init__(self, weight_bce=0.5, weight_dice=0.5):
        super(MixedLoss, self).__init__()
        self.bce = BCEWithLogitsLoss()
        self.dice = DiceLoss(include_background=True, to_onehot_y=True, sigmoid=True)
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        dice_loss = self.dice(inputs, targets)
        return self.weight_bce * bce_loss + self.weight_dice * dice_loss
##################################
# Variable to choose the dataset
dataset_choice = dataset_choice

# Define directories for datasets
data_dirs = {
    "2x": "/lustre/groups/iterm/Hazem/MA/data/2x",
    "4x": "/lustre/groups/iterm/Hazem/MA/data/4x"
}

# Set the data directory based on the chosen dataset
data_dir = data_dirs[dataset_choice]
##################################
#train_bg = sorted(glob.glob(os.path.join(data_dir, 'bg', "*.nii.gz")))[:2] #here I just used 2 patches of my data as a test so I shall not wait that long!
train_bg = sorted(glob.glob(os.path.join(data_dir, 'bg', "*.nii.gz")))
train_raw = sorted(glob.glob(os.path.join(data_dir, 'raw', "*.nii.gz")))
train_gt = sorted(glob.glob(os.path.join(data_dir, 'gt', "*.nii.gz")))
##################################

print(os.path.join(data_dir, 'bg', "*.nii.gz"))


# Print original paths #if this prints an empty list ([]), then the issue is definitely with the path or file presence.
original_bg_paths = glob.glob(os.path.join(data_dir, 'bg', "*.nii.gz"))
print("Original bg paths:")
for path in original_bg_paths:
    print(path)

# Print sorted paths
sorted_bg_paths = sorted(original_bg_paths)
print("Sorted bg paths:")
for path in sorted_bg_paths:
    print(path)


if train_bg:
    print(train_bg[0])
else:
    print("No .nii.gz files found in the specified directory.")
##################################

data_dicts = [
    {"bg": bg, "raw": raw, "label": gt}
    for bg, raw, gt in zip(train_bg, train_raw, train_gt)
]

print(data_dicts[0])
##################################
#Data Split
# Shuffle the data to ensure randomness BUT the testing dataset MUST ALWAYS be the SAME, so that I can compare between the performance of the different runs. Therefore, BEFORE shuffling, a random seed will be set for reproducibility
random.seed(42)

# Debugging: Print data_dicts before shuffling
print("Data before shuffling:")
for d in data_dicts:
    print(d)

# Shuffle the data to ensure randomness
random.shuffle(data_dicts)

# Debugging: Print data_dicts after shuffling
print("Data after shuffling:")
for d in data_dicts:
    print(d)


##################################
# Calculate the number of patches for each set
num_total = len(data_dicts)
num_test = max(1, round(num_total * 0.2))  #calculated as 20% of the total number of patches, rounded to the nearest whole number (ensuring at least one patch).
num_train_val = num_total - num_test #the remaining patches after allocating the test set.

# Split the remaining patches into training and validation
num_val = max(1, round(num_train_val * 0.2)) #calculated as 20% of the training/validation patches, rounded to the nearest whole number (ensuring at least one patch).
num_train = num_train_val - num_val # the remaining patches after allocating the validation set (and the testing set).

# Create the splits
test_files = data_dicts[:num_test]
train_val_files = data_dicts[num_test:]
train_files = train_val_files[:num_train]
val_files = train_val_files[num_train:]

# Print the split counts for verification
print(f"Total patches: {num_total}")
print(f"Training patches: {num_train}")
print(f"Validation patches: {num_val}")
print(f"Testing patches: {num_test}")


# Print to verify the contents of each split
print("Train files:", train_files)
print("Validation files:", val_files)
print("Test files:", test_files)

#Save the testing dataset to the Testing_Dataset folder
test_dataset_path = os.path.join(testing_dataset_dir, f"test_dataset_{slurm_job_id}_{dataset_choice}.json")
with open(test_dataset_path, 'w') as f: #Opens the file specified by test_dataset_path in write mode ('w').
    json.dump(test_files, f, indent=4)
print(f"Testing dataset saved to {test_dataset_path}")
##################################
#this function is a good check for my tensor dimensions, so that I do not have the dimensions-mismatch-ERROR!
#x is expected to be a dictionary containing keys for images and labels.
def print_shape(x):
    print(f"Shape of {x.keys()}: {x['image'].shape}, {x['label'].shape}") #This print statement is very useful for debugging. It ensures that the tensors have the expected dimensions at the point where the function is called.
    return x

##################################
#here am defining the dice_metric-class. It is NO MORE a function, it is a class, as MONAI-development team has RE-structured many of their features!
#It calculates the Dice coefficient, a measure of overlap between two samples.
#include_background=True: the metric will consider the background pixels in its calculation, which is useful in some cases where distinguishing background from the object of interest is important; which is EXACTLY my task.
#reduction="mean": indicates that the mean Dice score across all items in the batch will be calculated.
#Therefore, it provides a single scalar value representing the average Dice score over the batch, which is useful for tracking performance during training and validation.
#So here the class is defined and assigned to a variable. Then down at the evaluation block, the parameters (= my parameters: inputs & outputs) must be then added to the assigned variable.
dice_metric = DiceMetric(include_background=True, reduction="mean")
##################################

print("Create transforms")
#transformations will be applied in the VORGEGEBEN specific order!
#the order in the preprocessing is IMPORTANT! One of my mistakes was having the padding before the concatenating which caused errors! Furthermore, the normalization should be BEFORE the concatentation!
train_transforms = Compose( 
    [
        #Loads images and labels into tensors with shapes depending on the original data.
        LoadImaged(keys=["bg", "raw", "label"]), 
        
        #Medical images often come in formats where the channel dimension is absent, especially for grayscale images. Adding a channel dimension is necessary because deep learning frameworks like PyTorch expect a channel dimension.
        #It adds a channel dimension to the specified keys in the input dictionary. #'bg'/'raw'/'label' might become [1, D, H, W] if it was [D, H, W]. The added 1 indicates a single channel. 
        #Neural network architectures, including the 3D UNet, typically expect inputs with a shape that includes a channel dimension [C, D, H, W] or [C, H, W] for 2D images.
        #The UNet model expects input tensors with a specific shape that includes a channel dimension. Failing to add this dimension would result in shape mismatches and errors during model training or inference.
        AddChanneld(keys=["bg", "raw", "label"]),
        
        #It normalizes the intensity values of the specified keys in the input dictionary.
        #nonzero=True indicates that normalization should be computed only on non-zero values in the tensor. #CHANGE
        #Normalization typically involves scaling the intensity values of an image to a specific range, often [0, 1] or [-1, 1].
        #By focusing on non-zero values, the transform avoids skewing the normalization due to large regions of zeros (background).
        #Ensures that all images have a similar intensity range, which can improve the stability and performance of the neural network during training. AND
        #Helps in faster convergence of the model by providing a uniform range of input values. AND Reduces the impact of varying lighting conditions and contrast levels in the images.
        NormalizeIntensityd(keys="bg", nonzero=True), # Normalize intensity
        NormalizeIntensityd(keys="raw", nonzero=True), #Normalizes intensities but does not change tensor shapes.
        
        #dim=0 specifies that the concatenation should occur along the channel dimension (the first dimension in this case).
        #This concatenated image tensor is what will be fed into the neural network model during training and validation AND this Ensures that the model receives a multi-channel input, combining information from both bg and raw. #This can be particularly
        #useful in scenarios where different types of images provide complementary information. This combined input tensor allows the model to learn from both sources simultaneously, potentially improving segmentation performance.
        ConcatItemsd(keys=["bg", "raw"], name="image", dim=0), #Concatenates 'bg' and 'raw' along the channel dimension to form 'image'; 'image' becomes [2, D, H, W] where 2 comes from concatenating the channels of 'bg' and 'raw'.
        
        #spatial_size=(320, 320, 320) specifies the desired/target output size for each dimension (Depth, Height, Width).
        #mode='reflect' specifies the padding mode. 'reflect' mode pads with the reflection of the vector mirrored on the edge of the tensor.
        #Reflect padding helps in scenarios where the model needs context from the edges of the image. It provides a mirrored context, which can be beneficial for learning spatial features near the borders.
        #Check if the padding size (320, 320, 320) is suitable for your dataset and does not introduce too much background or alter the aspect ratio significantly.
        #If the input tensors are already larger than the specified spatial_size, no padding is added, and the tensors are left unchanged.
        #Ensures that all input tensors have a consistent size, which is crucial for batch processing and model input. This is particularly important for 3D medical images, where varying image sizes are common.
        #The UNet model expects input tensors of a specific shape, so padding is necessary to standardize the input sizes.
        SpatialPadd(keys=["image", "label"], spatial_size=(320, 320, 320), mode='reflect'), #Pads tensors to ensure all dimensions match (320, 320, 320); 'image' becomes [2, 320, 320, 320].; 'label' becomes [1, 320, 320, 320].
        
        #see explanation at the function
        Lambda(print_shape), #print the shape after padding as a CHECK for tensor dimensions, avoiding dimensions-mismatch! #print the shape after the 6 transforms
        
        #It converts the specified keys in the input dictionary to PyTorch tensors. PyTorch tensors are necessary for compatibility with the PyTorch framework, which is used for model training and inference.
        #Enables the use of GPU acceleration for training and inference, as PyTorch operations are optimized for performance on GPUs.
        ToTensord(keys=["image", "label"]),
        
        
        #you can check other "useful" transfoms from the safe-copy, vor allem: EnsureChannelFirstd (check the comment as well), cropping-transforms, random-tranforms, lambda-transforms #CHANGE
        #Same for evaluation transforms
    ]
)
val_transforms = Compose(
    [
        LoadImaged(keys=["bg", "raw", "label"]), 
        AddChanneld(keys=["bg", "raw", "label"]), 
        NormalizeIntensityd(keys="bg", nonzero=True), 
        NormalizeIntensityd(keys="raw", nonzero=True), 
        ConcatItemsd(keys=["bg", "raw"], name="image",dim=0), 
        SpatialPadd(keys=["image", "label"], spatial_size=(320, 320, 320), mode='reflect'), 
        Lambda(print_shape), 
        ToTensord(keys=["image", "label"]),
        #NOT every transform in the training-preprocessing shall be put as well in the evaluation-preprocessing!
    ]   
)

test_transforms = val_transforms  # Use the same transforms for validation and testing
##################################

print("Define dataset loaders")
#This dataset object is used to create a data loader that feeds batches of data into the model during training.
train_ds = Dataset(data=train_files, transform=train_transforms)

#The DataLoader efficiently handles the loading, preprocessing, and batching of data, making the training process smoother and faster.
#batch_size=1: The number of samples per batch. Setting it to 1 means each batch will contain one sample = only one "image" and only one "label".
#batch_size = 1 is useful for debugging, as it allows examining each sample individually. For actual training, larger batch sizes are typically used to take advantage of parallel processing and to stabilize gradient estimates.
#Increasing the number of batches allows the model to process multiple samples in parallel, improving training efficiency. AND Larger batch sizes can stabilize gradient estimates but require more memory.
#PASS AUF! You must know your data in order to set a reasonable optimal batch size! Otherwise, you will get a BIG error!
#shuffle=False: Whether to shuffle the data at every epoch. If False, data is not shuffled. If True, data is shuffled. #It is better to be true BUT how can I maintain the same testing-dataset while shuffling? #CHANGE
#For training, setting shuffle=True is generally recommended to ensure that the model does not learn the order of the data and to provide better generalization.
#num_workers=4: The number of subprocesses to use for data loading. More workers can speed up data loading. The optimal number of workers depends on the system’s hardware. #CHANGE
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=4) 

val_ds = Dataset(data=val_files, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=2)

test_ds = Dataset(data=test_files, transform=test_transforms)
test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=2)
##################################

# standard PyTorch program style: create UNet, DiceLoss and Adam optimizer
print("Create Model")
device = torch.device("cuda:0")
model = UNet(
    spatial_dims=3, #the input data is 3-dimensional (3D medical images).
    in_channels=2, #Specifies that the input has 2 channels (concatenated 'bg' and 'raw' images).
    out_channels=1, #Specifies that the output has 1 channel (a single segmentation mask) #try it with 2 #CHANGE (although it seems unlogic)
    
    #Specifies the number of filters (also known as feature maps or channels) in each layer of the UNet.
    #The specified number of filters impacts the model's capacity to learn features at different levels of abstraction.
    #Increasing the number of filters can improve the model's ability to capture complex features but also increases computational cost and memory usage.
    #These numbers define the depth (number of filters) of the convolutional layers at each stage of the encoder and decoder paths in the UNet.
    #e.g. Stage 1(Encoder): Input: Image with in_channels (e.g., 2 channels for 'bg' and 'raw').; Convolutions: 16 filters (specified by the first element of channels).; Operations: Convolution → Batch Normalization → Activation (ReLU).
    channels=(16, 32, 64, 128, 256),  
    
    #Strides refer to the step size with which the convolutional filters move across the input image. In the context of the UNet, strides specify the down-sampling rate for each layer in the encoder.
    #Down-sampling reduces the spatial dimensions (height, width, depth) of the feature maps while increasing the receptive field.
    #A stride of (2, 2, 2, 2) means that each down-sampling operation will reduce the size of the feature maps by a factor of 2. This is commonly implemented using operations like max pooling or convolutions with a stride greater than 1.
    #e.g. the input shape to the UNet is [2, 320, 320, 320] --> with a stride of 2 will reduce the dimensions to [16, 160, 160, 160]. -->  [32, 80, 80, 80]. --> [64, 40, 40, 40] --> [128, 20, 20, 20] --> [256, 10, 10, 10]
    strides=(2, 2, 2, 2), #Specifies the stride for each down-sampling layer.
    
    #Residual units refer to blocks of layers that use residual connections to facilitate training deeper networks. A residual connection adds the input of the block to its output, helping to mitigate the vanishing gradient problem and improving gradient flow through the network.
    #Typically, a residual block might include multiple convolutional layers, batch normalization, and ReLU activations. In this case, num_res_units=2 indicates that each block will consist of two such layers.
    #e.g. Without Residual Units: y = Conv(x); y = BatchNorm(y); y = ReLU(y) VS With Residual Units (A residual block with two units): residual = x; y = Conv(x); y = BatchNorm(y); y = ReLU(y); y = Conv(y); y = BatchNorm(y); y = y + residual; y = ReLU(y)
    #Residual connections help maintain gradients during backpropagation, making it easier to train deeper networks.
    #Can lead to better performance by allowing the network to learn identity mappings more easily, which is beneficial for deep architectures like UNet.
    num_res_units=2, # Specifies the number of residual units in each layer.
    
    norm=Norm.BATCH,
).to(device)
##################################
##################################
# Transfer Learning Code Starts Here
# Note: The segmentation model is defined and initialized before attempting to load any weights into it. The optimizer and loss function are defined after
# the model weights are updated, which is important because the optimizer needs to know which parameters to optimize (and their current states).
print("Transfer Learning Block begins!")
print("Load Pretrained Weights from Classification Model")

# path = path_to_classification_model_weights.pth #I chose really the best one --> testing score = 75% (Evaluation score = 83%)
path_class_best_metric_model = "/lustre/groups/iterm/Hazem/MA/Runs/run_27339286__Phase_2/best_metric_model_classification3d_array_27339286_50_0.001.pth" 

# Import the classification model
from Classification_3D import UNetForClassification

# Define and load the classification model
classification_model = UNetForClassification()
classification_model.load_state_dict(torch.load(path_class_best_metric_model)) #Loads the pretrained weights into the classification model(not yet to the segmentation one).
classification_model.to(device) #This ensures that any operations performed on the model (like weight copying) are done on the same device as the segmentation model.
classification_model.eval() #Sets the classification model to evaluation mode. This affects layers like dropout and batch normalization, which behave differently
# during training and evaluation. While we're not performing inference or training with the classification model, setting it to evaluation mode is a good practice to
# ensure consistent behavior if any operations depend on the model's mode. In other words, to disable layers like dropout and batch normalization from behaving differently.

# Get state dictionaries (parameter names AND tensors) of both models: Classification & Segmentation
classification_state_dict = classification_model.state_dict()
segmentation_state_dict = model.state_dict()

# Create a new state dictionary for the segmentation model
new_state_dict = {} #It serves as a container to store the parameters from the classification model that you want to transfer to the segmentation model after adjusting
#their names. By creating a new dictionary, you avoid altering the original state dictionaries of either model.
# You'll populate new_state_dict with parameters that match between the models after mapping the names. AND
# After populating, you'll update the segmentation model's state dictionary with new_state_dict.

#################################################################################

# Manipulating the first layer
# 1. Extract the First Layer Weights from the Classification Model:
class_first_layer_weights = classification_state_dict['unet.model.0.conv.weight']  # Shape: [out_channels, 1, k, k, k]

#Option A: Duplicate the Weights Across Channels
#Option A is generally preferred because it preserves the learned features and assumes both channels contribute similarly.
# 2. Duplicate the weights along the in_channels dimension
modified_first_layer_weights = class_first_layer_weights.repeat(1, 2, 1, 1, 1)  # Now shape: [out_channels, 2, k, k, k]

#OR

# #Option B: Average the Weights (If you believe the channels are similar, you can average the weights)
# modified_first_layer_weights = class_first_layer_weights / 2
# modified_first_layer_weights = torch.cat([modified_first_layer_weights, modified_first_layer_weights], dim=1)
# print("Option B in Action!")

# 3. Update the classification state dict with the modified weights
#Note: By updating the classification_state_dict, you ensure that when you proceed with your existing weight transfer loop, the first layer weights will be transferred correctly.
classification_state_dict['unet.model.0.conv.weight'] = modified_first_layer_weights

# # Map the parameter name to match the segmentation model
# seg_first_layer_name = 'model.0.conv.unit0.conv.weight'

# # Assign the modified weights
# segmentation_state_dict[seg_first_layer_name] = modified_first_layer_weights

print("First Layer adjusted!")

#####################################
#SAME for the ninth layer!
# 1. Extract the Ninth Layer Weights from the Classification Model:
class_ninth_layer_weights = classification_state_dict['unet.model.2.conv.weight']  # Shape: [32, 32, 3, 3, 3]

# Option A: Average the Weights Across Input Channels
# Since the segmentation model expects an input channel size of 1, you can average the weights across the input channels.
# 2. Average across the input channel dimension
modified_ninth_layer_weights = class_ninth_layer_weights.mean(dim=1, keepdim=True)  # Shape: [32, 1, 3, 3, 3]

# 3. Update the classification state dict with the modified weights
#Note: By updating the classification_state_dict, you ensure that when you proceed with your existing weight transfer loop, the first layer weights will be transferred correctly
classification_state_dict['unet.model.2.conv.weight'] = modified_ninth_layer_weights

print("Ninth Layer adjusted!")
#################################################################################



# Excluded parameters from the classification model
# excluded_params = [
#     # Encoder Layers
#     # 1st layer parameters (ALWAYS EXCLUDED!)
#     'unet.model.0.conv.weight',
#     'unet.model.0.conv.bias',
#     'unet.model.0.adn.A.weight',
#     # 2nd layer parameters
#     'unet.model.1.submodule.0.conv.bias',
#     'unet.model.1.submodule.0.adn.A.weight',
#     # 3rd layer parameters
#     'unet.model.1.submodule.1.submodule.0.conv.bias',
#     'unet.model.1.submodule.1.submodule.0.adn.A.weight',
#     # 4th layer parameters
#     'unet.model.1.submodule.1.submodule.1.submodule.0.conv.bias',
#     'unet.model.1.submodule.1.submodule.1.submodule.0.adn.A.weight',

#     # Bottleneck
#     # 5th layer parameters
#     'unet.model.1.submodule.1.submodule.1.submodule.1.submodule.conv.weight',
#     'unet.model.1.submodule.1.submodule.1.submodule.1.submodule.conv.bias',
#     'unet.model.1.submodule.1.submodule.1.submodule.1.submodule.adn.A.weight',
    
#     # Decoder Layers
#     # 6th layer parameters
#     'unet.model.1.submodule.1.submodule.1.submodule.2.conv.weight',
#     'unet.model.1.submodule.1.submodule.1.submodule.2.conv.bias',
#     'unet.model.1.submodule.1.submodule.1.submodule.2.adn.A.weight',
#     # 7th layer parameters
#     'unet.model.1.submodule.1.submodule.2.conv.weight',
#     'unet.model.1.submodule.1.submodule.2.conv.bias',
#     'unet.model.1.submodule.1.submodule.2.adn.A.weight',
#     # 8th layer parameters
#     'unet.model.1.submodule.2.conv.weight',
#     'unet.model.1.submodule.2.conv.bias',
#     'unet.model.1.submodule.2.adn.A.weight',
#     # 9th layer parameters (ALWAYS EXCLUDED!) (there is no adn layer here!!)
#     'unet.model.2.conv.weight',
#     'unet.model.2.conv.bias',
    
#     # Classification Layer
#     'fc.weight',
#     'fc.bias',
# ]


excluded_params = [
    # Encoder Layers
    # 1st layer parameters (ALWAYS EXCLUDED!) (implemented in the below if condition, NOPE! here is better!)
    # 2nd layer parameters
    # 3rd layer parameters
    # 4th layer parameters

    # Bottleneck
    # 5th layer parameters
    
    # Decoder Layers
    # 6th layer parameters
    'unet.model.1.submodule.1.submodule.1.submodule.2.conv.weight',
    'unet.model.1.submodule.1.submodule.1.submodule.2.conv.bias',
    'unet.model.1.submodule.1.submodule.1.submodule.2.adn.A.weight',
    # 7th layer parameters
    'unet.model.1.submodule.1.submodule.2.conv.weight',
    'unet.model.1.submodule.1.submodule.2.conv.bias',
    'unet.model.1.submodule.1.submodule.2.adn.A.weight',
    # 8th layer parameters
    'unet.model.1.submodule.2.conv.weight',
    'unet.model.1.submodule.2.conv.bias',
    'unet.model.1.submodule.2.adn.A.weight',
    # 9th layer parameters (ALWAYS EXCLUDED!) (implemented in the below if condition, NOPE! here is better!) (there is no adn layer here!!)
    'unet.model.2.conv.weight',
    'unet.model.2.conv.bias',
    # Classification Layer (excluded via 'fc' in name)

]

# Initialize a counter for the number of transferred parameters
transfer_count = 0

# Map and transfer the convolutional weights
for name, param in classification_state_dict.items():
    # Skip EXPLICITLY excluded parameters
    if name in excluded_params:
        continue 
    # Skip the first convolutional layer due to input channel mismatch, and the NINTH layer due to output channel mismatch
    # if name == 'unet.model.0.conv.weight' or name == 'unet.model.2.conv.weight':
    #     continue
    # Skip biases, normalization layers, and fully connected layer
    #if 'bias' in name or 'adn' in name or 'fc' in name:
    #if 'adn' in name or 'fc' in name:
    if 'fc' in name:
        continue
     

    # Map parameter names (adapting the classification name to the segmentation one, NOT vice versa!)
    seg_name = name.replace('unet.model.', 'model.')

    if 'adn' in seg_name:
        # Handle 'adn' parameters separately
        if 'submodule.2' in seg_name or 'submodule.1.submodule.2' in seg_name or 'submodule.1.submodule.1.submodule.2' in seg_name:
            # Decoder layers
            seg_name = seg_name.replace('.adn.', '.0.adn.')
        else:
            # Encoder and bottleneck layers
            seg_name = seg_name.replace('.adn.', '.conv.unit0.adn.')
    else:
        # Handle convolutional weights and biases
        if 'submodule.2' in seg_name or 'submodule.1.submodule.2' in seg_name or 'submodule.1.submodule.1.submodule.2' in seg_name:
            # Decoder layers
            seg_name = seg_name.replace('.conv.', '.0.conv.')
        else:
            # Encoder and bottleneck layers
            seg_name = seg_name.replace('.conv.', '.conv.unit0.conv.')
    
    # # Check if the parameter is in the decoder layers
    # if 'submodule.2' in seg_name or 'submodule.1.submodule.2' in seg_name or 'submodule.1.submodule.1.submodule.2' in seg_name:
    #     # For decoder layers, insert '.0' before '.conv.unit0.conv'
    #     seg_name = seg_name.replace('.conv.', '.0.conv.')
        
    # else:
    #     # For encoder layers, use the existing mapping
    #     seg_name = seg_name.replace('.conv.', '.conv.unit0.conv.')
    
    # # seg_name = name.replace('unet.model.', 'model.')
    # # seg_name = seg_name.replace('.conv.', '.conv.unit0.conv.')

    #For debugging: Print original and mapped names
    print(f"Original name: {name}, Mapped name: {seg_name}")

    # Check if the parameter exists in the segmentation model
    if seg_name in segmentation_state_dict: #Only transfer parameters where the names match.
        if param.shape == segmentation_state_dict[seg_name].shape: #Only transfer parameters where the shapes are identical.
            new_state_dict[seg_name] = param
            print(f"Transferring {name} --> {seg_name}")
            transfer_count += 1  # Increment the counter
        else:
            print(f"Shape mismatch: {name} ({param.shape}) vs {seg_name} ({segmentation_state_dict[seg_name].shape}), skipping.")
    else:
        print(f"{seg_name} not found in segmentation model, skipping.") 

# After the loop, print the total number of transferred parameters
print(f"Total number of parameters transferred: {transfer_count}")

# Update the segmentation model's state dictionary 
segmentation_state_dict.update(new_state_dict) #Updates the segmentation model's state dictionary with the transferred parameters.
model.load_state_dict(segmentation_state_dict, strict=False) #This line loads the updated segmentation_state_dict (which now includes the transferred weights) into the segmentation model.
#Use strict=False to avoid errors due to missing keys or unexpected keys. --> model.load_state_dict(segmentation_state_dict, strict=False)
#Setting strict=False allows the model to load the state dict even if not all keys match. HOWEVER, this method may raise an error if there are unmatched keys, which is
#likely given that you're not transferring all weights.
#WARNING: strict=False MUST be included to avoid errors due to missing keys, especially since you're not transferring all weights.
#when you are not initializing (transferring) all parameters in the segmentation model, setting strict=False in model.load_state_dict() is necessary.
#strict=False: Allows the loading process to ignore missing or unexpected keys. && This is suitable when you're only loading a subset of the model's parameters (e.g., when
#doing transfer learning and not transferring all weights).

#Freeze the transferred layers #HOWEVER, I will comment this whole tiny block at first. Maybe later, I will test the script with freezing
for name, param in model.named_parameters(): #Ensure that the parameter names in model.named_parameters() match those in new_state_dict.
    if name in new_state_dict:
        param.requires_grad = False #you prevent the optimizer from updating these parameters during training.
        print(f"Freezing layer: {name}")

# print out the names and shapes of the transferred parameters to verify the transfer and ensure correctness:
print("Transferred Parameters:")
for name in new_state_dict:
    print(f"{name}: {new_state_dict[name].shape}")


# Transfer Learning Code Ends Here

##################################
##################################
print("Create Loss")

#include_background=True: Includes the background class in the loss calculation. For binary segmentation tasks, this parameter ensures that the loss accounts for both foreground and background pixels.
#Typically for DiceLoss with a binary label you would set include_background to True since we want to compare the foreground against background.
#to_onehot_y=True: Converts the ground truth labels to one-hot encoding before computing the loss. Ensures that each class is treated independently in the loss calculation. For binary segmentation, the ground truth labels are converted from shape [B, 1, D, H, W] to [B, C, D, H, W], where B is the batch size and C is the number of classes.
#to_onehot_y == OneHotEncoding ?? ((OneHotEncoding = Turn all unique values into lists of 0's and 1's where the target value is 1 and the rest are 0's. For example, with car colors green, red and blue: a green car's color feature would be represented as [1,0,0] while a red one would be [0,1,0]))
#OneHotEncoding is a type of feature encoding which is turning values within your dataset(even images) into numbers, bec. ML-model requires all values to be numerical
#sigmoid=True: Applies a sigmoid activation to the network outputs before computing the loss. The sigmoid function maps the network outputs to the range [0, 1], which is suitable for binary and multi-class segmentation tasks. Converts logits (raw output values from the network) into probabilities.
#You can NOT apply DiceMetric instead of DiceLoss. DiceMetric is NOT designed to calculate the losses, bus as a metric for the evaluation!
#loss_function = DiceLoss(include_background=True, to_onehot_y=True, sigmoid=True) #MONAI-DiceLoss #try softmax #CHANGE
#loss_function = BCEWithLogitsLoss() #PyTorch - binary crossentropy loss COMBINED with a sigmoid layer --> more numerically stable
loss_function = DiceCELoss(include_background=True, to_onehot_y=True, sigmoid=True) #MONAI-DiceCELoss #try softmax #CHANGE #it is NOT PURELY binary cross entropy loss
#loss_function = MixedLoss(weight_bce=0.5, weight_dice=0.5) #PyTorch & MONAI - MIXED loss: 0.5


loss_function_name = "DiceCELoss"

##################################
print("Create Optimizer ")
#The learning rate is a crucial hyperparameter that controls how much to adjust the model's weights with respect to the loss gradient during each optimization step.
#A smaller learning rate can lead to more stable training but might require more epochs to converge.
learning_rate = learning_rate

# Initialize an empty set to store unique learning rates
unique_learning_rates = set()

#model.parameters() provides the optimizer with the parameters of the model that need to be updated during training.
#After defining the optimizer, add the scheduler definition
optimizer = torch.optim.Adam(model.parameters(), learning_rate)


#RLOP reduces the learning rate when a metric has stopped improving. This is useful because it allows the model to converge more effectively by lowering the learning rate when the training seems to stagnate.
#Using RLOP during validation ensures that the learning rate is reduced when the model's performance plateaus, allowing for finer adjustments based on actual validation performance.
#optimizer: The optimizer instance whose learning rate needs to be adjusted.
#'min': Mode for the metric; 'min' means the learning rate will be reduced when the monitored quantity (e.g., validation loss) has stopped decreasing.
#patience=5: Number of epochs with no improvement after which the learning rate will be reduced.
#factor=0.5: Factor by which the learning rate will be reduced. new_lr = old_lr * factor.
#verbose=True: If True, prints a message to stdout for each update. --> so I do NOT really need the print-statements!

#cosinA implements a learning rate schedule that follows the cosine function, gradually reducing the learning rate over time.
#T_max: After T_max iterations, the learning rate will reach its minimum value (eta_min) and then the cycle can repeat if restarts are enabled (though restarts are not part of the basic CosineAnnealingLR and would require using CosineAnnealingWarmRestarts instead).
#Example: If T_max is set to 50, the learning rate will decrease from its initial value to eta_min following a cosine curve over 50 epochs.
#eta_min: Definition: Minimum learning rate. Purpose: This is the lowest value the learning rate will reach during the annealing process. It ensures that the learning rate does not drop to zero, which can be undesirable as it would stop the learning process completely.
#verbose: If set to True, this will print messages indicating the learning rate adjustments. This can be helpful for debugging and understanding how the learning rate is changing over time.  If verbose is True, you will see output messages showing the learning rate changes at each step.

#cosinAWarmRestarts provides periodic restarts which can help the model escape local minima and explore new regions of the loss landscape during training. This can be particularly beneficial in maintaining model momentum.
#T_0: Number of epochs for the first restart
#T_mult: A factor that increases T_0 after each restart.
#eta_min: Minimum learning rate after annealing.
print("Create Scheduler")

#scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

# Define scheduler based on the JSON configuration
if chosen_scheduler:
    if chosen_scheduler["type"] == "cosinA":
        scheduler = CosineAnnealingLR(optimizer, T_max=chosen_scheduler["params"]["T_max"], 
                                                   eta_min=chosen_scheduler["params"]["eta_min"],
                                                   verbose=chosen_scheduler["params"]["verbose"])
    elif chosen_scheduler["type"] == "RLOP":
        scheduler = ReduceLROnPlateau(optimizer, mode=chosen_scheduler["params"]["mode"],
                                                   patience=chosen_scheduler["params"]["patience"], 
                                                   factor=chosen_scheduler["params"]["factor"], 
                                                   verbose=chosen_scheduler["params"]["verbose"])
    elif chosen_scheduler["type"] == "cosinAWR":
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=chosen_scheduler["params"]["T_0"],
                                                T_mult=chosen_scheduler["params"]["T_mult"],
                                                eta_min=chosen_scheduler["params"]["eta_min"],
                                                verbose=chosen_scheduler["params"]["verbose"])
else:
    scheduler = None


if scheduler and isinstance(scheduler, (torch.optim.lr_scheduler.CosineAnnealingLR, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts)):
    print(f"The {scheduler_info} will be used in the training loop!")
else:
    print("No Scheduler will be used in the training loop!")

if scheduler and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
    print(f"The {scheduler_info} will be used in the validation loop!")
else:
    print("No Scheduler will be used in the validation loop!")

#GradScaler is used for mixed precision training, which allows for faster computation and reduced memory usage by using 16-bit (half-precision) floating-point numbers instead of the default 32-bit (single-precision).
#GradScaler scales the loss before backpropagation to prevent gradients from becoming too small (underflow) or too large (overflow) in the 16-bit representation.
#The scaler dynamically adjusts these scales, ensuring that gradients are neither too small to cause underflow nor too large to cause overflow.
scaler = torch.cuda.amp.GradScaler() #Scaled gradient
##################################
print("Execute a typical PyTorch training process")
print("Dry Tesl ALL DONE")
max_epochs = max_epochs #only 10 as a test
val_interval = val_interval #from 2 to 1
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []
post_pred = AsDiscrete(argmax=True, to_onehot=2, n_classes=2) # argmax turns values into discretes #2 classes: background and foreground #to_onehot=n_classes
post_label = AsDiscrete(to_onehot=2, n_classes=2) 
#both post_pred and post_label are eliminated = NOT used!!
##################################
#Clear CUDA cache to free up memory before starting the training loop
#This ensures that any residual memory from previous operations is freed up.
#This setup should help in mitigating the out-of-memory error.
import torch
torch.cuda.empty_cache()


for epoch in range(max_epochs):
    print("-" * 40)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train() # Tells the model that it's being trained and not used for inference, model.eval()
    epoch_loss = 0
    step = 0

    for batch_data in train_loader: #how big is the batch? #For each batch, computes the loss, backpropagates gradients, and updates the model weights.
        step += 1
        inputs, labels = (
            batch_data["image"].to(device),
            batch_data["label"].to(device),
        )
        print(f"Input shape before model: {inputs.shape}")  #to check input size
        optimizer.zero_grad() #Before you compute gradients for a new batch, you need to zero out the gradients from the previous batch. If you don't zero them out, gradients from different batches will accumulate, NOT desired!!


        #Mixed Precision Training: Utilizes GPU capabilities for faster and memory-efficient training.
        with torch.cuda.amp.autocast(): # (automated mixed precision) #allowing performance in a lower precision --> requires less memory, thus: speeding up the training process!
            outputs = model(inputs)
            #loss = loss_function(outputs, labels) #for DiecLoss
            loss = loss_function(outputs, labels.float()) # BCEWithLogitsLoss expects both outputs (already is float) and labels to be of floating-point type. --> labels.float()
        scaler.scale(loss).backward() #check scaler?!
        scaler.step(optimizer)
        scaler.update()
        
        epoch_loss += loss.item() #accumulates the total loss over the epoch
        print(
            f"{step}/{len(train_ds) // train_loader.batch_size}, "
            f"train_loss: {loss.item():.4f}")

    epoch_loss /= step
    epoch_loss_values.append(epoch_loss) #Logs the average loss for each epoch.
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    # Update the learning rate using the scheduler if available
    #The reason for this if-check is that ReduceLROnPlateau scheduler steps based on validation metrics rather than epoch count, so we don't want to call scheduler.step() for ReduceLROnPlateau in the training loop.
    #Only for logging the learning rates of schedulers that step per epoch --> CosineAnnealingLR steps based on epoch, so it should be called at the end of each epoch in the training loop.
    if scheduler and isinstance(scheduler, (torch.optim.lr_scheduler.CosineAnnealingLR, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts)): #you can use OR as well, but you will then need a nested if which I don't want
        scheduler.step()
        #scheduler.get_last_lr(): Returns a list of the last computed learning rates by the scheduler. This is a list because PyTorch optimizers can have multiple parameter groups, each with its own learning rate.
        #[0]: Assumes that there is only one parameter group in the optimizer. If you have multiple parameter groups, you would need to handle each one appropriately.
        current_lr = scheduler.get_last_lr()[0]  # Assuming only one param group 
        print(f"Current Learning Rate after epoch {epoch + 1}: {current_lr}")
        unique_learning_rates.add(current_lr)

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad(): #disabling gradient calculation
            metric_sum = 0.0
            metric_count = 0
            for val_data in val_loader:
                val_inputs, val_labels = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                )
                roi_size = (300,300,300) #adapt it to the dimensions of my data
                sw_batch_size = 4 #number of slices or batches processed simultaneously in the sliding window inference.
                
                with torch.cuda.amp.autocast(): #check #sliding_window_inference VS conventional_inference: val_outputs = model(val_inputs)
                    val_outputs = model(val_inputs)

                    # Apply sigmoid activation and threshold to obtain binary outputs
                    val_outputs = torch.sigmoid(val_outputs)  # Sigmoid to convert logits to probabilities
                    val_outputs = (val_outputs > 0.5).float()  # Thresholding probabilities to binary values
                    #I think I need here to visualize my outputs as a CHECK, before and after!
                    val_labels[val_labels > 0] = 1

    
                #to ensure data shapes and types match the expected:
                print("Validation outputs shape:", val_outputs.shape)
                print("Validation labels shape:", val_labels.shape)
                print("Validation outputs unique values:", torch.unique(val_outputs))
                print("Validation labels unique values:", torch.unique(val_labels))


                #Computes Dice metrics on validation data.
                dice_metric(y_pred=val_outputs, y=val_labels)
                #"dice_metric" is used to evaluate the model's performance on the validation set. It calculates the Dice coefficient between the predicted segmentation masks (val_outputs) and the ground truth masks/labels (val_labels).
                
                
            
            #This line computes the average Dice score across all batches processed so far.
            #This call computes the mean Dice score over all batches, providing a single performance metric for the epoch.
            metric = dice_metric.aggregate().item() #this line is giving us the average DiceScore for the epoch without having to manually add the individual scores then dividing them by the length/Abzahl of the whole
            #Resets the metric computation to start fresh for the next epoch.
            dice_metric.reset()
            metric_values.append(metric)
            
            #Logs and saves the best model based on validation performance
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                best_model_path = os.path.join(my_models_dir, f"best_metric_model_unet_{slurm_job_id}_{dataset_choice}_{learning_rate}_{max_epochs}.pth")
                torch.save(model.state_dict(), best_model_path)
                print("Saved new best metric model at:", best_model_path)
            print(
                f"current epoch: {epoch + 1} current mean dice: {metric:.10f}"
                f"\nbest mean dice: {best_metric:.10f} "
                f"at epoch: {best_metric_epoch}"
            )

        #In the validation loop, after computing the validation metric, update the scheduler based on validation metric!
        #The ReduceLROnPlateau scheduler should be applied in the validation loop because it adjusts the learning rate based on the validation performance, not on the epoch count.
        #if scheduler and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        if scheduler and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(metric)
            current_lr = scheduler.optimizer.param_groups[0]['lr']
            print(f"Current Learning Rate after validation at epoch {epoch + 1}: {current_lr}")
            unique_learning_rates.add(current_lr)

    #plots the training loss and validation Dice metrics over epochs.
    print("Creating the Plots!")
    if (epoch + 1) % 1 == 0: # from %3 to %1, to see more!
        print("Plot the loss and metric")
        
        plt.figure("train", (12, 6))
        plt.subplot(1, 2, 1)
        plt.title("Epoch Average Loss")
        x = [i + 1 for i in range(len(epoch_loss_values))]
        y = epoch_loss_values
        plt.xlabel("epoch")
        plt.plot(x, y)
        
        plt.subplot(1, 2, 2)
        plt.title("Val Mean Dice")
        x = [val_interval * (i + 1) for i in range(len(metric_values))]
        y = metric_values
        plt.xlabel("epoch")
        plt.plot(x, y)

        #saves the plots to specified directories.
        fname = os.path.join(my_plots_dir, f"metrics_unet_{slurm_job_id}_{dataset_choice}_{learning_rate}_{max_epochs}.png")
        plt.savefig(fname, dpi=300, facecolor='w', edgecolor='w',
                format='png', transparent=False, pad_inches=0.1)
        plt.show()
        print(f"Plot saved to {fname}")
        
        #saves the latest model state to specified directories.
        latest_model_path = os.path.join(my_models_dir, f"latest_metric_model_unet_{slurm_job_id}_{dataset_choice}_{learning_rate}_{max_epochs}.pth")
        torch.save(model.state_dict(), latest_model_path)
        print(f"Saved the latest model state to {latest_model_path}")

# Print the unique learning rates used
print("-" * 40)
print("-" * 40)
if scheduler:
    print("Unique Learning Rates used during training:", sorted(unique_learning_rates))
else:
    print("No learning rate scheduling applied, constant learning rate used.")


#prints a summary message with the best metric achieved and the corresponding epoch.
print("-" * 40)
print(
    f"train completed, best_metric: {best_metric:.4f} "
    f"at epoch: {best_metric_epoch}") 
##################################
# Evaluate on the test dataset
print("-" * 40)
print("FINAL STEP: Evaluate on test dataset")
model.eval()
with torch.no_grad(): #disabling gradient calculation
    test_metric = DiceMetric(include_background=True, reduction="mean") #defining the test_metric-CLASS (NOT an object)
    #for test_data in test_loader:
    for i, test_data in enumerate(test_loader): #to keep track of which file is being processed
        test_inputs, test_labels = (
            test_data["image"].to(device),
            test_data["label"].to(device),
        )

        with torch.cuda.amp.autocast(): #check #sliding_window_inference VS conventional_inference: val_outputs = model(val_inputs)
            test_outputs = model(test_inputs)
            #Apply sigmoid activation and threshold to obtain binary outputs
            test_outputs = torch.sigmoid(test_outputs) # Sigmoid to convert logits to probabilities
            test_outputs = (test_outputs > 0.5).float() # Thresholding probabilities to binary values
            test_labels[test_labels > 0] = 1

        # Iterate over the batch and save each output individually
        for j in range(test_outputs.shape[0]): #iterates over each item in the batch (test_outputs.shape[0])
            # Extract original file path to generate the new file name
            original_file_path = test_files[i * test_outputs.shape[0] + j]["raw"]
            original_file_name = os.path.basename(original_file_path)
            new_file_name = original_file_name.replace("raw", "test").replace("bg", "test").replace("gt", "test")

            # Convert tensors to numpy arrays
            output_array = test_outputs.cpu().numpy()[j, 0] #The output_array is generated for each item in the batch, and then saved individually.

            # Check and resize if necessary
            raw_image_path = test_files[i * test_outputs.shape[0] + j]["raw"]
            raw_image = nib.load(raw_image_path).get_fdata()
            if output_array.shape != raw_image.shape:
                zoom_factors = np.array(raw_image.shape) / np.array(output_array.shape)
                output_array = zoom(output_array, zoom_factors, order=1) # order=1 for bilinear interpolation


            # Save output as NIfTI file
            output_nifti_path = os.path.join(output_nifti_dir, new_file_name)
            output_nifti = nib.Nifti1Image(output_array, np.eye(4))
            nib.save(output_nifti, output_nifti_path)
        
        
            #to ensure data shapes and types match the expected:
            print(f"Test output {j} shape:", test_outputs[j].shape)
            print(f"Test label {j} shape:", test_labels[j].shape)
            print(f"Test output {j} unique values:", torch.unique(test_outputs[j]))
            print(f"Test label {j} unique values:", torch.unique(test_labels[j]))

            #Computes Dice metrics on testing data.
            test_metric(y_pred=test_outputs, y=test_labels)

    test_metric_value = test_metric.aggregate().item() #This line computes the average Dice score across all batches processed so far.
    test_metric.reset() ##Resets the metric computation to start fresh for the next epoch.
    print(f"Test Mean Dice: {test_metric_value:.10f}")


print(f"NIfTI files saved to {output_nifti_dir}")
print("-" * 40)
##################################
# Prepare data for CSV using Pandas
epochs = list(range(1, max_epochs + 1))
training_losses = epoch_loss_values
validation_metrics = [None] * max_epochs

for i, metric in enumerate(metric_values):
    validation_epoch = (i + 1) * val_interval
    validation_metrics[validation_epoch - 1] = metric

# Create a DataFrame
df = pd.DataFrame({
    "Epoch": epochs,
    "Average Training Loss": training_losses,
    "Average Validation Dice Metric": validation_metrics
})

# Add rows for best metric and test metric
# Create DataFrames for the additional metrics
best_metric_df = pd.DataFrame({
    "Epoch": ["Best Evaluation Metric"],
    "Average Training Loss": [best_metric],
    "Average Validation Dice Metric": [None]
})

test_metric_df = pd.DataFrame({
    "Epoch": ["Test Dice Metric"],
    "Average Training Loss": [test_metric_value],
    "Average Validation Dice Metric": [None]
})

# Concatenate the DataFrames
df = pd.concat([df, best_metric_df, test_metric_df], ignore_index=True)

# Create a self-explanatory CSV file name
csv_file_name = f"tracking_{slurm_job_id}_{dataset_choice}_{learning_rate}_{max_epochs}_{test_metric_value:.4f}.csv"
csv_file_path = os.path.join(tracking_dir, csv_file_name)



# Save the DataFrame to a CSV file
df.to_csv(csv_file_path, index=False)

print(f"Metrics CSV saved to {csv_file_path}")
##################################
## the Runs folder - all in one
# Save csv file at Runs
run_csv_file_path = os.path.join(run_dir, csv_file_name)
df.to_csv(run_csv_file_path, index=False)
print(f"Metrics CSV saved to {csv_file_path} and {run_csv_file_path}")

# Save the best model at Runs
run_best_model_path = os.path.join(run_dir, f"best_metric_model_unet_{slurm_job_id}_{dataset_choice}_{learning_rate}_{max_epochs}.pth")
shutil.copy(best_model_path, run_best_model_path)
print(f"Saved best model at: {best_model_path} and {run_best_model_path}")

# Save the latest model state at Runs
run_latest_model_path = os.path.join(run_dir, f"latest_metric_model_unet_{slurm_job_id}_{dataset_choice}_{learning_rate}_{max_epochs}.pth")
shutil.copy(latest_model_path, run_latest_model_path)
print(f"Saved latest model state to {latest_model_path} and {run_latest_model_path}")

# Save the plot at Runs
run_plot_file_path = os.path.join(run_dir, f"metrics_unet_plot_{slurm_job_id}_{dataset_choice}_{learning_rate}_{max_epochs}.png")
shutil.copy(fname, run_plot_file_path)
print(f"Plot saved to {fname} and {run_plot_file_path}")      

# Save the test dataset at Runs
run_test_dataset_path = os.path.join(run_dir, f"test_dataset_{slurm_job_id}_{dataset_choice}.json")
shutil.copy(test_dataset_path, run_test_dataset_path)
print(f"Testing dataset saved to {test_dataset_path} and {run_test_dataset_path}")

# Save output and error at Runs
slurm_output_file = f"/lustre/groups/iterm/Hazem/MA/HPC/slurm_outputs/3D_Seg_{slurm_job_id}_output.txt"
slurm_error_file = f"/lustre/groups/iterm/Hazem/MA/HPC/slurm_outputs/3D_Seg_{slurm_job_id}_error.txt"
run_slurm_output_file = os.path.join(run_dir, f"3D_Seg_{slurm_job_id}_0_output.txt")
run_slurm_error_file = os.path.join(run_dir, f"3D_Seg_{slurm_job_id}_0_error.txt")

shutil.copy(slurm_output_file, run_slurm_output_file)
shutil.copy(slurm_error_file, run_slurm_error_file)
print(f"Slurm output file copied to {run_slurm_output_file}")
print(f"Slurm error file copied to {run_slurm_error_file}")
##################################
#final print
print("-" * 40)
###################################'
end_time = time.time()

elapsed_time = end_time - start_time

hours, rem = divmod(elapsed_time, 3600)
minutes, seconds = divmod(rem, 60)
print(f"Total Execution Time: {int(hours):02d}h {int(minutes):02d}m {seconds:.2f}s")
print("-" * 40)
##################################
print("ALL DONE!") 