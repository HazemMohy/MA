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
from monai.data import CacheDataset, list_data_collate, decollate_batch, DataLoader
import torch
import os
import glob
import numpy as np
import argparse
import shutil
import json

# parse the parameters from json file
with open('param.json') as json_file:
    config = json.load(json_file)

# args parser
parser = argparse.ArgumentParser()
parser.add_argument("--job", type=str, required=True)
args = parser.parse_args()

# Data directory
data_dir = "/lustre/groups/iterm/Annotated_Datasets/Annotated Datasets/Alpha-BTX - Neuromuscular Junctions/2x"

# Define the device
device = torch.device("cuda:0")

# Define the PyTorch Lightning module
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
        ).to(device)
        self.loss_function = DiceLoss(to_onehot_y=True, sigmoid=True)
        self.post_pred = Compose([EnsureType("tensor", device="cpu"), AsDiscrete(argmax=True, to_onehot=None)])
        self.post_label = Compose([EnsureType("tensor", device="cpu"), AsDiscrete(to_onehot=None)])
        self.dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
        self.best_val_dice = 0
        self.best_val_epoch = 0
        self.validation_step_outputs = []

    def forward(self, x):
        return self._model(x)

    def prepare_data(self):
        train_raw = sorted(glob.glob(os.path.join(data_dir, 'raw', "*.nii.gz")))
        train_bg = sorted(glob.glob(os.path.join(data_dir, 'bg', "*.nii.gz")))
        train_gt = sorted(glob.glob(os.path.join(data_dir, 'gt', "*.nii.gz")))
        data_dicts = [{"bg": bg, "raw": raw, "label": gt} for bg, raw, gt in zip(train_raw, train_bg, train_gt)]
        train_files, val_files = data_dicts[:-9], data_dicts[-9:]
        set_determinism(seed=0)
        transforms = Compose([
            LoadImaged(keys=["raw", "bg", "label"]),
            EnsureChannelFirstd(keys=["raw", "bg", "label"]),
            SpatialPadd(keys=["raw", "bg", "label"], spatial_size=(320, 320, 320), mode='reflect'),
            ConcatItemsd(keys=["raw", "bg"], name="image", dim=0),
            NormalizeIntensityd(keys="image", nonzero=True),
            Lambdad(keys='label', func=lambda x: (x >= 0.5).astype(np.float32)),
            Lambdad(keys='image', func=lambda x: (x/x.max()).astype(np.float32)),
        ])
        self.train_ds = CacheDataset(data=train_files, transform=transforms, cache_rate=1.0, num_workers=4)
        self.val_ds = CacheDataset(data=val_files, transform=transforms, cache_rate=1.0, num_workers=4)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=4, shuffle=False, num_workers=4, collate_fn=list_data_collate)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=1, num_workers=4)

    def configure_optimizers(self):
        return torch.optim.Adam(self._model.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        output = self.forward(images)
        loss = self.loss_function(output, labels)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        outputs = self.forward(images)
        loss = self.loss_function(outputs, labels)
        outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
        labels = [self.post_label(i) for i in decollate_batch(labels)]
        self.dice_metric(y_pred=outputs, y=labels)
        return {"val_loss": loss, "val_number": len(outputs)}

    def on_validation_epoch_end(self):
        mean_val_dice = self.dice_metric.aggregate().item()
        self.dice_metric.reset()
        if mean_val_dice > self.best_val_dice:
            self.best_val_dice = mean_val_dice
            self.best_val_epoch = self.current_epoch
        self.validation_step_outputs.clear()

# Debugging data loading
def debug_data_loading(loader):
    for i, data in enumerate(loader):
        print(f"Batch {i + 1}")
        print(f"Images shape: {data['image'].shape}, Labels shape: {data['label'].shape}")
        print(f"Images max: {data['image'].max()}, Images min: {data['image'].min()}")
        print(f"Labels max: {data['label'].max()}, Labels min: {data['label'].min()}")
        if i == 0:  # Only print one batch for brevity
            break

# Set up the model and data
model = Net()
model.prepare_data()
train_loader = model.train_dataloader()
val_loader = model.val_dataloader()

# Debugging
print("Debugging Training Data Loader:")
debug_data_loading(train_loader)
print("\nDebugging Validation Data Loader:")
debug_data_loading(val_loader)

# Setup logger and trainer
log_dir = os.path.join("/lustre/groups/iterm/Hazem/MA/HPC/logs/", args.job)
os.makedirs(log_dir, exist_ok=True)
tb_logger = pytorch_lightning.loggers.TensorBoardLogger(save_dir=log_dir)
trainer = pytorch_lightning.Trainer(
    devices=[0],
    max_epochs=10,
    logger=tb_logger,
    enable_checkpointing=True,
    num_sanity_val_steps=1,
    log_every_n_steps=5,
)

# Start training
trainer.fit(model)
print(f"Training completed, best_metric: {model.best_val_dice:.4f} at epoch {model.best_val_epoch}")
shutil.make_archive(log_dir, 'zip', log_dir)
