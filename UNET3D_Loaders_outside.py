import pytorch_lightning
from monai.utils import set_determinism
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    EnsureType,
    NormalizeIntensityd,
    Lambdad,
    ConcatItemsd,
    SpatialPadd,
    ToTensord,
)
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, list_data_collate, decollate_batch, DataLoader
from monai.config import print_config
from monai.apps import download_and_extract
import torch
import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import glob
import numpy as np
import json
import argparse
import shutil
import torchvision.utils as vutils
print("Importing Done!")

# parse the parameters from json file
with open('param.json') as json_file:
    config = json.load(json_file)

# args parser
parser = argparse.ArgumentParser()
parser.add_argument("--job", type=str, required=True)
args = parser.parse_args()
print("Parsing parameters Done!")

# Data directory
data_dir= "/lustre/groups/iterm/Hazem/MA/data/2x"
print("Path found Done!")

#add device
#device = torch.device("cuda:0")

# Define the lighting module
class Net(pytorch_lightning.LightningModule):
    def __init__(self):
        super().__init__()
        self._model = UNet(
            spatial_dims=3,
            in_channels=2,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
        )#.to(device) #device added
        #########
        #self.loss_function = DiceLoss(to_onehot_y=False, softmax=True)
        self.loss_function = DiceLoss(softmax=True)
        #self.loss_function = bce(to_onehot_y=False, sigmoid=True) #switched from the softmax function to the sigmoid one
        #self.loss_function = bce(to_onehot_y=False, sigmoid=True) #switched from the softmax function to the sigmoid one
        self.post_pred = Compose([EnsureType("tensor", device="cpu"), AsDiscrete(argmax=True)]) #to_onehot?
        self.post_label = Compose([EnsureType("tensor", device="cpu")]) #to_onehot deleted
        self.dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False) #Typically for DiceLoss with a binary label you would set include_background to True since we want to compare the foreground against background

        #self.loss_function = DiceLoss(to_onehot_y=True, softmax=True)
        #self.post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
        #self.post_label = Compose([AsDiscrete(to_onehot=2)])
        #self.dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
        
        #https://github.com/Project-MONAI/MONAI/discussions/1727
        ##########
        self.best_val_dice = 0
        self.best_val_epoch = 0
        self.validation_step_outputs = []
        self.train_ds = 0
        self.val_ds = 0
        #prepare_data()

    def forward(self, x):
        return self._model(x)

    def prepare_data(self):
        # set up the correct data path and create a list of the names of the nifti Voxels
        train_raw = sorted(glob.glob(os.path.join(data_dir, 'raw', "*.nii.gz")))
        train_bg = sorted(glob.glob(os.path.join(data_dir, 'bg', "*.nii.gz")))
        train_gt = sorted(glob.glob(os.path.join(data_dir, 'gt', "*.nii.gz")))

        # create a dictionary for image and label correspondance
        data_dicts = [
            {"bg": bg, "raw": raw, "label": gt} for bg, raw, gt in zip(train_bg, train_raw, train_gt)
        ]
        # training files are all the Voxels but the last 2, the validation files are the last two .nii files
        train_files, val_files = data_dicts[:-2], data_dicts[-2:] #total of my files = 5


        # set deterministic training for reproducibility
        set_determinism(seed=0)
        # define the data transforms
        train_transforms = Compose(
            [
                LoadImaged(keys=["raw", "bg", "label"]),
                EnsureChannelFirstd(keys=["raw","bg", "label"]), # (Channel_dim,X_dim,Y_dim,Z_dim): tensor size = torch.unsqueeze(0)
                SpatialPadd(keys=["raw","bg", "label"], spatial_size=(320, 320, 320), mode='reflect'), # added reflective padding #Padding: Check if the padding size (320, 320, 320) is suitable for your dataset and does not introduce too much background or alter the aspect ratio significantly.
                ConcatItemsd(keys=["raw", "bg"], name="image", dim=0), # stacks bg and raw into a tensor with 2 channels
                NormalizeIntensityd(
                    keys = "image",
                    nonzero = True,
                ), # Normalization values between 0 and 1 #Normalization: Ensure that it correctly scales the image intensities as expected by your model.
                Lambdad(
                    keys='label', 
                    func=lambda x: (x >= 0.5).astype(np.float32) # nicht größer, sondern größer gleich!!!!
                    ), # threshhold opration for the binray mask either 1 or 0
                Lambdad(
                    keys='image',
                    func=lambda x: (x/x.max()).astype(np.float32) #to resolve the invalid values-issue (only 0's and 1's)
                ),
                ToTensord(keys=["image", "label"]),
            ])
        val_transforms = Compose(
            [
                LoadImaged(keys=["raw", "bg", "label"]),
                EnsureChannelFirstd(keys=["raw","bg", "label"]), # (Channel_dim,X_dim,Y_dim,Z_dim): tensor size = torch.unsqueeze(0)
                SpatialPadd(keys=["raw","bg", "label"], spatial_size=(320, 320, 320), mode='reflect'),
                ConcatItemsd(keys=["raw", "bg"], name="image", dim=0),
                NormalizeIntensityd(
                    keys = "image",
                    nonzero = True,
                ),
                Lambdad(
                    keys='label', 
                    func=lambda x: (x >= 0.5).astype(np.float32)
                    ),
                Lambdad(
                    keys='image',
                    func=lambda x: (x/x.max()).astype(np.float32)
                ),
                ToTensord(keys=["image", "label"]),
            ])

        # we use cached datasets - these are 10x faster than regular datasets but succeptible to RAM overflow
        
        self.train_ds = CacheDataset(
            data=train_files,
            transform=train_transforms,
            cache_rate=1.0,
            num_workers=4,
        )
        self.val_ds = CacheDataset(
            data=val_files,
            transform=val_transforms,
            cache_rate=1.0,
            num_workers=4,
        )
        

    #   self.train_ds = monai.data.Dataset(
    #             data=train_files, transform=train_transforms)
    #   self.val_ds = monai.data.Dataset(
    #             data=val_files, transform=val_transforms)

    '''
    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_ds,
            batch_size=1, #increased to 4
            shuffle=False,
            num_workers=4,
            collate_fn=list_data_collate,
        )
        return train_loader
    

    def val_dataloader(self):
        val_loader = DataLoader(self.val_ds, batch_size=1, num_workers=4)
        return val_loader
    '''

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self._model.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"] #images, labels, outputs (the next line) abspeichern und auf Laptop schauen
        output = self.forward(images)
        loss = self.loss_function(output, labels)
        tensorboard_logs = {"train_loss": loss.item()} #später!!!
        return {"loss": loss, "log": tensorboard_logs}


    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        #roi_size = (160, 160, 160)
        #sw_batch_size = 4
        #outputs = sliding_window_inference(images, roi_size, sw_batch_size, self.forward) #nur mit self.forward(), genau wie training_step!!!!
        outputs = self.forward(images)
        loss = self.loss_function(outputs, labels) #outputs_and_labels_before_pred
        outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
        labels = [self.post_label(i) for i in decollate_batch(labels)] #outputs_and_labels_after_pred_abspeichern_und_schauen_UND_in in the last epochs, not in the first ones!
        self.dice_metric(y_pred=outputs, y=labels)
        d = {"val_loss": loss, "val_number": len(outputs)}
        self.validation_step_outputs.append(d)
        return d



    def on_validation_epoch_end(self):
        val_loss, num_items = 0, 0
        for output in self.validation_step_outputs:
            val_loss += output["val_loss"].sum().item()
            num_items += output["val_number"]
        mean_val_dice = self.dice_metric.aggregate().item()
        self.dice_metric.reset()
        mean_val_loss = torch.tensor(val_loss / num_items)
        tensorboard_logs = {
            "val_dice": mean_val_dice,
            "val_loss": mean_val_loss,
        }
        if mean_val_dice > self.best_val_dice:
            self.best_val_dice = mean_val_dice
            self.best_val_epoch = self.current_epoch
        print(
            f"current epoch: {self.current_epoch} "
            f"current mean dice: {mean_val_dice:.4f}"
            f"\nbest mean dice: {self.best_val_dice:.4f} "
            f"at epoch: {self.best_val_epoch}"
            f"at epoch: {val_loss}" #added validation loss zu printen
            f"\nValidation loss: {mean_val_loss}"  # Correctly use the calculated mean_val_loss
            
        )
        self.validation_step_outputs.clear()  # free memory
        return {"log": tensorboard_logs}




# initialise the LightningModule
net = Net()
net.prepare_data()  # Ensure datasets are prepared and assigned to net.train_ds and net.val_ds

#defining train_ds and val_ds outside the class as well, not anymore!!!!!

#get the loaders outside the class
def train_dataloader(train_ds): #net.train_ds instead of self. Nope, I will initialize it seperately at the end!
    train_loader = DataLoader(
        train_ds,
        batch_size=4, #increased to 4
        shuffle=False,
        num_workers=4,
        collate_fn=list_data_collate,
    )
    return train_loader
    

def val_dataloader(val_ds): #net.val_ds instead of self. Nope, I will initialize it seperately at the end!
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=4)
    return val_loader


# Create the dataloaders using datasets prepared in net
train_loader = train_dataloader(net.train_ds)
val_loader = val_dataloader(net.val_ds)


# Debugging data loading
def debug_data_loading(loader):
    for i, data in enumerate(loader):
        print(f"Batch {i + 1}")
        print(f"Images shape: {data['image'].shape}, Labels shape: {data['label'].shape}")
        print(f"Images max: {data['image'].max()}, Images min: {data['image'].min()}")
        print(f"Labels max: {data['label'].max()}, Labels min: {data['label'].min()}")
        if i == 0:  # Only print one batch for brevity
            break

# Call the Debugging data loading function
print("Debugging Training Data Loader:")
debug_data_loading(train_loader)
print("Debugging Validation Data Loader:")
debug_data_loading(val_loader)


# set up loggers and checkpoints
log_dir = os.path.join("/lustre/groups/iterm/Hazem/MA/HPC/logs/", args.job)
os.makedirs(log_dir, exist_ok=True)
tb_logger = pytorch_lightning.loggers.TensorBoardLogger(save_dir=log_dir)

# initialise Lightning's trainer.
trainer = pytorch_lightning.Trainer(
    devices=[0],
    max_epochs=7,
    logger=tb_logger,
    enable_checkpointing=True,
    num_sanity_val_steps=0, #zu Null gesetzt
    log_every_n_steps=1,
)


# START training
trainer.fit(net, train_dataloaders=train_loader, val_dataloaders=val_loader)
print(f"train completed, best_metric: {net.best_val_dice:.4f} " f"at epoch {net.best_val_epoch}")


# Archive logs
shutil.make_archive(log_dir, 'zip', log_dir)
