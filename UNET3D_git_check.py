print("Hello World!")

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
from monai.metrics import compute_meandice
from monai.losses import DiceLoss, TverskyLoss, DiceCELoss
from monai.inferers import sliding_window_inference
from monai.data import SmartCacheDataset, CacheDataset, DataLoader, Dataset
from monai.config import print_config
from monai.apps import download_and_extract
from monai.optimizers import Novograd
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import sys
import matplotlib
import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import glob
from glob import glob

import warnings
warnings.filterwarnings("ignore")  # remove some scikit-image warnings
print_config()


root_dir = "//homes/lindholm/monai/data/nets20"
CT_images = sorted(glob("//homes/lindholm/monai/data/nets20/images/*0000.nii.gz"))
PET_images  = sorted(glob("//homes/lindholm/monai/data/nets20/images/*0001.nii.gz"))
train_labels  = sorted(glob("//homes/lindholm/monai/data/nets20/labels/*.nii.gz"))
  
data_dict = [
    {"pet": PET_images, "ct": CT_images, "label": label_name}
    for PET_images, CT_images, label_name in zip(PET_images, CT_images, train_labels)
]

print(data_dict[0])

train_files = data_dict[:-9]
val_files = data_dict[-9:]

print("Create transformers")
train_transforms = Compose(
    [
        LoadImaged(keys=["pet", "ct", "label"]),
        AddChanneld(keys=["pet", "ct", "label"]),
        NormalizeIntensityd(keys="pet", nonzero=True), # Normalize intensity
        NormalizeIntensityd(keys="ct", nonzero=True), # Normalize intensity
        ConcatItemsd(keys=["pet", "ct"], name="image", dim=0),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(256,256,256),
            pos=5, # I 1 ud af 1+1 vil crop have en label voxel som centrum
            neg=1,
            num_samples=3,
            image_key="image",
            image_threshold=0,
        ),
        
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0), # Random flip image on axis
        RandAffined(
            keys=['image', 'label'],
            mode=('bilinear', 'nearest'),
            prob=0.2,
            shear_range=(0, 0, 0.1),
            rotate_range=(0, 0, np.pi/16),
            scale_range=(0.1, 0.1, 0.1)),
        ToTensord(keys=["image", "label"]),
    ]
)
val_transforms = Compose(
    [
        LoadImaged(keys=["pet", "ct", "label"]),
        AddChanneld(keys=["pet", "ct", "label"]),
        NormalizeIntensityd(keys="pet", nonzero=True),
        NormalizeIntensityd(keys="ct", nonzero=True),
        ConcatItemsd(keys=["pet", "ct"], name="image",dim=0),
        
        ToTensord(keys=["image", "label"]),
    ]   
)


print("Define dataset loaders")
train_ds = Dataset(data=train_files, transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4)

val_ds = Dataset(data=val_files, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=2)

print("Create Model, Loss, Optimizer")
# standard PyTorch program style: create UNet, DiceLoss and Adam optimizer
device = torch.device("cuda:0")
model = UNet(
    dimensions=3,
    in_channels=2,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
).to(device)

squared_pred=True, reduction='mean', batch=False)
loss_function = DiceLoss(include_background=False, to_onehot_y=True, softmax=True)

learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), learning_rate)
scaler = torch.cuda.amp.GradScaler() #Scaled gradient 


print("Execute a typical PyTorch training process")
max_epochs = 10000
val_interval = 2
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []
post_pred = AsDiscrete(argmax=True, to_onehot=True, n_classes=2) # argmax turns values into discretes
post_label = AsDiscrete(to_onehot=True, n_classes=2)

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
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(): # (automated mixed precision)
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        epoch_loss += loss.item()
        print(
            f"{step}/{len(train_ds) // train_loader.batch_size}, "
            f"train_loss: {loss.item():.4f}")
       
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            metric_sum = 0.0
            metric_count = 0
            for val_data in val_loader:
                val_inputs, val_labels = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                )
                roi_size = (256, 256,256)
                sw_batch_size = 4
                
                with torch.cuda.amp.autocast():
                    val_outputs = sliding_window_inference(
                        val_inputs, roi_size, sw_batch_size, model, overlap=0.25,
                    )

                val_outputs = post_pred(val_outputs)
                val_labels = post_label(val_labels)
                value = compute_meandice(
                    y_pred=val_outputs,
                    y=val_labels,
                    include_background=False,
                )
                metric_count += len(value)
                metric_sum += value.sum().item()
                
            metric = metric_sum / metric_count
            metric_values.append(metric)
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(
                    root_dir, "best_metric_model_unet.pth"))
                print("saved new best metric model")
            print(
                f"current epoch: {epoch + 1} current mean dice: {metric:.10f}"
                f"\nbest mean dice: {best_metric:.10f} "
                f"at epoch: {best_metric_epoch}"
            )
            
    
    if (epoch + 1) % 3 == 0:
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
        
        plt.show()
        fname = "//homes/lindholm/monai/data/nets20/metrics_unet.png"
        plt.savefig(fname, dpi=300, facecolor='w', edgecolor='w',
                    format='png', transparent=False, pad_inches=0.1,
        )
    
        torch.save(model.state_dict(), os.path.join(
                    root_dir, "latest_metric_model_unet.pth"))
            
print(
    f"train completed, best_metric: {best_metric:.4f} "
    f"at epoch: {best_metric_epoch}")  