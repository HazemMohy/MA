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
    SpatialPadd
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

# parse the parameters from json file
with open('param.json') as json_file:
    config = json.load(json_file)

# Data directory
data_dir= "/lustre/groups/iterm/Annotated_Datasets/Annotated Datasets/Alpha-BTX - Neuromuscular Junctions/2x"

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
        )
        #########
        self.loss_function = DiceLoss(to_onehot_y=True, softmax=True)
        self.post_pred = Compose([EnsureType("tensor", device="cpu"), AsDiscrete(argmax=True, to_onehot=1)]) #1 nicht 2!!
        self.post_label = Compose([EnsureType("tensor", device="cpu"), AsDiscrete(to_onehot=1)])
        self.dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        ##########
        self.best_val_dice = 0
        self.best_val_epoch = 0
        self.validation_step_outputs = []

    def forward(self, x):
        return self._model(x)

    def prepare_data(self):
        # set up the correct data path and create a list of the names of the nifti Voxels
        train_raw = sorted(glob.glob(os.path.join(data_dir, 'raw', "*.nii.gz")))
        train_bg = sorted(glob.glob(os.path.join(data_dir, 'bg', "*.nii.gz")))
        train_gt = sorted(glob.glob(os.path.join(data_dir, 'gt', "*.nii.gz")))

        # create a dictionary for image and label correspondance
        data_dicts = [
            {"bg": bg, "raw": raw, "label": gt} for bg, raw, gt in zip(train_raw, train_bg, train_gt)
        ]
        # training files are all the Voxels but the last 2, the validation files are the last nine .nii files
        train_files, val_files = data_dicts[:-2], data_dicts[-2:]

        # set deterministic training for reproducibility
        set_determinism(seed=0)
        # define the data transforms
        train_transforms = Compose(
            [
                LoadImaged(keys=["raw", "bg", "label"]),
                EnsureChannelFirstd(keys=["raw","bg", "label"]), # (Channel_dim,X_dim,Y_dim,Z_dim): tensor size = torch.unsqueeze(0)
                SpatialPadd(keys=["raw","bg", "label"], spatial_size=(320, 320, 320), mode='reflect'), # added reflective padding
                ConcatItemsd(keys=["raw", "bg"], name="image", dim=0), # stacks bg and raw into a tensor with 2 channels
                NormalizeIntensityd(
                    keys = "image",
                    nonzero = True,
                ), # Normalization values between 0 and 1
                Lambdad(
                    keys='label', 
                    func=lambda x: (x >= 0.5).astype(np.float32) # nicht größer, sondern größer gleich!!!!
                    ), # threshhold opration for the binray mask either 1 or 0
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

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_ds,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            collate_fn=list_data_collate,
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(self.val_ds, batch_size=1, num_workers=4)
        return val_loader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self._model.parameters(), lr=1e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        output = self.forward(images)
        loss = self.loss_function(output, labels)
        tensorboard_logs = {"train_loss": loss.item()} #später!!!
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        roi_size = (160, 160, 160)
        sw_batch_size = 4
        outputs = sliding_window_inference(images, roi_size, sw_batch_size, self.forward) #nur mit self.forward(), genau wie training_step!!!!
        loss = self.loss_function(outputs, labels)
        outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
        labels = [self.post_label(i) for i in decollate_batch(labels)]
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
        )
        self.validation_step_outputs.clear()  # free memory
        return {"log": tensorboard_logs}


# Training
# initialise the LightningModule
net = Net()

# set up loggers and checkpoints
log_dir = os.path.join("/home/iterm/hazem.abdel-rehim/MA", "logs")
tb_logger = pytorch_lightning.loggers.TensorBoardLogger(save_dir=log_dir)

# initialise Lightning's trainer.
trainer = pytorch_lightning.Trainer(
    devices=[0],
    max_epochs=7,
    logger=tb_logger,
    enable_checkpointing=True,
    num_sanity_val_steps=1,
    log_every_n_steps=2,
)

# train
trainer.fit(net)

print(f"train completed, best_metric: {net.best_val_dice:.4f} " f"at epoch {net.best_val_epoch}")


