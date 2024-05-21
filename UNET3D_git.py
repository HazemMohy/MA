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
torch.multiprocessing.set_sharing_strategy('file_system') #????????????
import sys
import matplotlib
import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import glob
##################################

import warnings
warnings.filterwarnings("ignore")  # remove some scikit-image warnings
#print_config()
##################################
slurm_job_id = os.environ.get('SLURM_JOB_ID', 'default_job_id')

my_models_dir = "/lustre/groups/iterm/Hazem/MA/models"
os.makedirs(my_models_dir, exist_ok=True)

my_plots_dir = "/lustre/groups/iterm/Hazem/MA/plots"
os.makedirs(my_plots_dir, exist_ok=True)



##################################
#data_dir = "/lustre/groups/iterm/Annotated_Datasets/Annotated Datasets/Alpha-BTX- NeuromuscularJunctions/2x" # work with 4x
#data_dir = "/lustre/groups/iterm/Annotated_Datasets/Annotated Datasets/Alpha-BTX - Neuromuscular Junctions/2x/"
data_dir = "/lustre/groups/iterm/Hazem/MA/data/2x"


train_bg = sorted(glob.glob(os.path.join(data_dir, 'bg', "*.nii.gz")))[:2]
train_raw = sorted(glob.glob(os.path.join(data_dir, 'raw', "*.nii.gz")))[:2]
train_gt = sorted(glob.glob(os.path.join(data_dir, 'gt', "*.nii.gz")))[:2]


print(os.path.join(data_dir, 'bg', "*.nii.gz"))
print(glob.glob(os.path.join(data_dir, 'bg', "*.nii.gz"))) #if this prints an empty list ([]), then the issue is definitely with the path or file presence.
if train_bg:
    print(train_bg[0])
else:
    print("No .nii.gz files found in the specified directory.")


data_dicts = [
    {"bg": bg, "raw": raw, "label": gt}
    for bg, raw, gt in zip(train_bg, train_raw, train_gt)
]

print(data_dicts[0])
##################################

train_files = data_dicts[:-1]
val_files = data_dicts[-1:] #total of my files = 5 (2)
##################################


print("Create transforms")
train_transforms = Compose(
    [
        LoadImaged(keys=["bg", "raw", "label"]),
        #EnsureChannelFirstd(keys=["raw","bg", "label"]), # (Channel_dim,X_dim,Y_dim,Z_dim): tensor size = torch.unsqueeze(0)
        AddChanneld(keys=["bg", "raw", "label"]),
        #SpatialPadd(keys=["raw","bg", "label"], spatial_size=(320, 320, 320), mode='reflect'), # added reflective padding #Padding: Check if the padding size (320, 320, 320) is suitable for your dataset and does not introduce too much background or alter the aspect ratio significantly.
        NormalizeIntensityd(keys="bg", nonzero=True), # Normalize intensity
        NormalizeIntensityd(keys="raw", nonzero=True), # Normalize intensity
        ConcatItemsd(keys=["bg", "raw"], name="image", dim=0),
        #CropForegroundd(keys=["image", "label"], source_key="image"),
        
        #RandCropByPosNegLabeld(
        #   keys=["image", "label"],
        #    label_key="label",
        #    spatial_size=(256,256,256),
        #    pos=5, # I 1 ud af 1+1 vil crop have en label voxel som centrum
        #    neg=1,
        #    num_samples=3,
        #    image_key="image",
        #    image_threshold=0,
        #),
        
        #RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0), # Random flip image on axis
        
        #RandAffined(
        #   keys=['image', 'label'],
        #   mode=('bilinear', 'nearest'),
        #    prob=0.2,
        #    shear_range=(0, 0, 0.1),
        #    rotate_range=(0, 0, np.pi/16),
        #    scale_range=(0.1, 0.1, 0.1)),'''
        
        #Lambdad(
        #        keys='label', 
        #        func=lambda x: (x >= 0.5).astype(np.float32) # nicht größer, sondern größer gleich!!!!
        #        ), # threshhold opration for the binray mask either 1 or 0
        #Lambdad(
        #        keys='image',
        #        func=lambda x: (x/x.max()).astype(np.float32) #to resolve the invalid values-issue (only 0's and 1's)
        #        ),'''
        ToTensord(keys=["image", "label"]),
    ]
)
val_transforms = Compose(
    [
        LoadImaged(keys=["bg", "raw", "label"]),
        #EnsureChannelFirstd(keys=["raw","bg", "label"]), # (Channel_dim,X_dim,Y_dim,Z_dim): tensor size = torch.unsqueeze(0)
        #SpatialPadd(keys=["raw","bg", "label"], spatial_size=(320, 320, 320), mode='reflect'),
        AddChanneld(keys=["bg", "raw", "label"]),
        NormalizeIntensityd(keys="bg", nonzero=True),
        NormalizeIntensityd(keys="raw", nonzero=True),
        ConcatItemsd(keys=["bg", "raw"], name="image",dim=0),
        
        #Lambdad(
        #    keys='label', 
        #    func=lambda x: (x >= 0.5).astype(np.float32)
        #    ),
        #Lambdad(
        #    keys='image',
        #    func=lambda x: (x/x.max()).astype(np.float32)
        #    ),'''
        ToTensord(keys=["image", "label"]),
    ]   
)
##################################

print("Define dataset loaders")
train_ds = Dataset(data=train_files, transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=1, shuffle=False, num_workers=4) #set shuffle to False and inspect!

val_ds = Dataset(data=val_files, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=2) #set shuffle to False and inspect!
##################################


# standard PyTorch program style: create UNet, DiceLoss and Adam optimizer
print("Create Model")
device = torch.device("cuda:0")
model = UNet(
    spatial_dims=3,
    in_channels=2,
    out_channels=1, #try it with 2
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
).to(device)
##################################
print("Create Loss")
#squared_pred=True, reduction='mean', batch=False) #here is sth. wrong!! GOOGLE!
loss_function = DiceLoss(include_background=True, to_onehot_y=True, sigmoid=True) #set include_backgroung to true; and switch from softmax to sigmoid; check as well the to_onehot_y-parameter #try softmax again
#to_onehot_y == OneHotEncoding ?? ((OneHotEncoding = Turn all unique values into lists of 0's and 1's where the target value is 1 and the rest are 0's. For example, with car colors green, red and blue: a green car's color feature would be represented as [1,0,0] while a red one would be [0,1,0]))
#OneHotEncoding is a type of feature encoding which is turning values within your dataset(even images) into numbers, bec. ML-model requires all values to be numerical
#loss_function = DiceMetric(include_background=True, reduction="mean", get_not_nans=False) #try DiceMetric instead of DiceLoss or Nah!
#Typically for DiceLoss with a binary label you would set include_background to True since we want to compare the foreground against background. However, check it again!
##################################
print("Create Optimizer ")
learning_rate = 1e-5
optimizer = torch.optim.Adam(model.parameters(), learning_rate)
scaler = torch.cuda.amp.GradScaler() #Scaled gradient
#The scaler dynamically adjusts these scales, ensuring that gradients are neither too small to cause underflow nor too large to cause overflow.
##################################
print("Execute a typical PyTorch training process")
max_epochs = 10 #only 10 as a test
val_interval = 1 #from 2 to 1
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []
post_pred = AsDiscrete(argmax=True, to_onehot=True, n_classes=2) # argmax turns values into discretes #2 classes: background and foreground #try to_onehot=2
#post_pred = Compose([Activations(sigmoid=True), AsDiscrete(argmax=True, to_onehot=True, n_classes=2)]) #try this!!
post_label = AsDiscrete(to_onehot=True, n_classes=2) #argmax may be needed here as well #try to_onehot=2
#post_pred and post_label will now act as transforms!
##################################

for epoch in range(max_epochs):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train() # Tells the model that it's being trained and not used for inference, model.eval()
    epoch_loss = 0
    step = 0

    for batch_data in train_loader:
        step += 1
        inputs, labels = (
            batch_data["image"].to(device),
            batch_data["label"].to(device),
        )
        optimizer.zero_grad() #Before you compute gradients for a new batch, you need to zero out the gradients from the previous batch. If you don't zero them out, gradients from different batches will accumulate, NOT desired!!

        with torch.cuda.amp.autocast(): # (automated mixed precision) #allowing performance in a lower precision --> requires less memory, thus: speeding up the training process!
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
        scaler.scale(loss).backward() #check scaler?!
        scaler.step(optimizer)
        scaler.update()
        
        epoch_loss += loss.item() #accumulates the total loss over the epoch
        print(
            f"{step}/{len(train_ds) // train_loader.batch_size}, "
            f"train_loss: {loss.item():.4f}")


    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")


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
                
                with torch.cuda.amp.autocast(): #check #sliding_window_inference VS conventional_inference: val_outputs = model(val_inputs
                    val_outputs = sliding_window_inference(
                        val_inputs, roi_size, sw_batch_size, model, overlap=0.25,
                    )

                val_outputs = post_pred(val_outputs) #clarify post_pred
                val_labels = post_label(val_labels) #clarify post_label
                value = DiceMetric(
                    y_pred=val_outputs,
                    y=val_labels,
                    include_background=True, #include_background shall be set to True! #try dice_metric instead of compute_meandice for validation
                )
                #value = DiceMetric(include_background=True, reduction="mean") # For validation #try dice_metric instead of compute_meandice for validation
                #value = DiceMetric(include_background=True, reduction="mean", get_not_nans=False) # For validation #try dice_metric instead of compute_meandice for validation
                metric_count += len(value) #to compute the average later
                metric_sum += value.sum().item() #sum of ALL DiceScores
                
            metric = metric_sum / metric_count #average DiceScore for the epoch
            metric_values.append(metric)
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                best_model_path = os.path.join(my_models_dir, f"best_metric_model_unet_{slurm_job_id}.pth")
                torch.save(model.state_dict(), best_model_path)
                print("Saved new best metric model at:", best_model_path)
            print(
                f"current epoch: {epoch + 1} current mean dice: {metric:.10f}"
                f"\nbest mean dice: {best_metric:.10f} "
                f"at epoch: {best_metric_epoch}"
            )


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

        fname = os.path.join(my_plots_dir, f"metrics_unet_{slurm_job_id}.png")
        plt.savefig(fname, dpi=300, facecolor='w', edgecolor='w',
                format='png', transparent=False, pad_inches=0.1)
        plt.show()
        print(f"Plot saved to {fname}")
        
        latest_model_path = os.path.join(my_models_dir, f"latest_metric_model_unet_{slurm_job_id}.pth")
        torch.save(model.state_dict(), latest_model_path)
        print(f"Saved the latest model state to {latest_model_path}")


print(
    f"train completed, best_metric: {best_metric:.4f} "
    f"at epoch: {best_metric_epoch}") 


