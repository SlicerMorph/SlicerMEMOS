
# Cell 2
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use only GPU 0
import shutil
import tempfile

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import torch

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    RandAffined,
    RandGaussianNoised,
    EnsureChannelFirstd,
    EnsureTyped,
)

from monai.config import print_config
from monai.metrics import DiceMetric
from monai.networks.nets import UNETR

from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)

print_config()

device = torch.device("cuda:0")
print(f"Using device: {device}")

# Cell 4
directory = "/data/hps/home/amaga/magalab/user/amaga/MEMOS_retrain"
root_dir = tempfile.mkdtemp() if directory is None else directory
print(root_dir)

# Cell 6
image_dim = 128
pixel_dim = 1
train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(
            keys=["image", "label"],
            pixdim=(pixel_dim, pixel_dim, pixel_dim),
            mode=("bilinear", "nearest"),
        ),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=0,
            a_max=255,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(image_dim, image_dim, image_dim),
            pos=1,
            neg=1,
            num_samples=4,
            image_key="image",
            image_threshold=0,
        ),
        RandRotate90d(
            keys=["image", "label"],
            prob=0.10,
            max_k=3,
        ),
        RandShiftIntensityd(
            keys=["image"],
            offsets=0.10,
            prob=0.50,
        ),
        RandGaussianNoised(
            keys=['image', 'label'],
            prob=0.1, 
            mean=0,
            std=.1,
        ),
        RandAffined(
            keys=['image', 'label'],
            mode=('bilinear', 'nearest'),
            prob=0.1, 
            spatial_size=(image_dim, image_dim, image_dim),
            rotate_range=(0, 0, np.pi/15),
            scale_range=(0.1, 0.1, 0.1),
        ),
        EnsureTyped(keys=["image", "label"]),
    ]
)
val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(
            keys=["image", "label"],
            pixdim=(pixel_dim, pixel_dim, pixel_dim),
            mode=("bilinear", "nearest"),
        ),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=0,
            a_max=255,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        EnsureTyped(keys=["image", "label"]),
    ]
)

# Cell 7
split_JSON = "/data/hps/home/amaga/magalab/user/amaga/MEMOS_retrain/dataset_KOMP_baseline_edited.json"
datalist = load_decathlon_datalist(split_JSON, True, "training")
val_files = load_decathlon_datalist(split_JSON, True, "testing")
train_ds = CacheDataset(
    data=datalist,
    transform=train_transforms,
    cache_num=45,
    cache_rate=1.0,
    num_workers=8,
)
train_loader = DataLoader(
    train_ds, batch_size=1, shuffle=True, num_workers=8, pin_memory=True
)
val_ds = CacheDataset(
    data=val_files, transform=val_transforms, cache_num=11, cache_rate=1.0, num_workers=4
)
val_loader = DataLoader(
    val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True
)

# Cell 11
torch.set_num_threads(24)

model = UNETR(
    in_channels=1,
    out_channels=51,
    img_size=(image_dim, image_dim, image_dim),
    feature_size=16,
    hidden_size=768,
    mlp_dim=3072,
    num_heads=12,
    proj_type="perceptron",
    norm_name="instance",
    conv_block=True,
    res_block=True,
    dropout_rate=0.0,
).to(device)

loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
torch.backends.cudnn.benchmark = True
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

# Cell 12 - Define post-processing transforms
post_label = AsDiscrete(to_onehot=51)
post_pred = AsDiscrete(argmax=True, to_onehot=51)

# Cell 13
def validation():
    """Validation function"""
    model.eval()
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    
    print("[Validation] Starting...", flush=True)
    dice_vals = []
    
    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            print(f"  Processing case {idx+1}/11...", flush=True)
            val_inputs = batch["image"].to(device)
            val_labels = batch["label"].to(device)
            
            val_outputs = sliding_window_inference(
                val_inputs, (image_dim, image_dim, image_dim), 4, model
            )
            
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [post_label(x) for x in val_labels_list]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [post_pred(x) for x in val_outputs_list]
            
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
            dice = dice_metric.aggregate().item()
            dice_vals.append(dice)
            dice_metric.reset()
        
    mean_dice = np.mean(dice_vals)
    print(f"[Validation] Done. Mean Dice: {mean_dice:.4f}", flush=True)
    return mean_dice


def train_epoch(epoch):
    """Train for one epoch"""
    model.train()
    epoch_loss = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", dynamic_ncols=True)
    
    for batch_idx, batch in enumerate(pbar):
        x, y = batch["image"].to(device), batch["label"].to(device)
        
        optimizer.zero_grad()
        logit_map = model(x)
        loss = loss_function(logit_map, y)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        pbar.set_description(f"Epoch {epoch} (loss={loss.item():.5f})")
    
    avg_loss = epoch_loss / len(train_loader)
    return avg_loss


# Cell 14
max_iterations = 10000
eval_num = 45  # Validate every 45 steps (every epoch with single GPU)

best_dice = 0.0
best_step = 0
epoch_loss_values = []
metric_values = []

# Cell 15 - Main training loop
num_epochs = max_iterations // len(train_loader) + 1

for epoch in range(num_epochs):
    avg_loss = train_epoch(epoch)
    global_step = (epoch + 1) * len(train_loader)
    
    epoch_loss_values.append(avg_loss)
    print(f"\nEpoch {epoch} complete. Avg loss: {avg_loss:.4f}, Global step: {global_step}")
    
    # Validate if we've hit eval_num steps or finished training
    should_validate = (global_step % eval_num == 0) or (global_step >= max_iterations)
    if should_validate:
        mean_dice = validation()
        metric_values.append(mean_dice)
        
        if mean_dice > best_dice:
            best_dice = mean_dice
            best_step = global_step
            torch.save(
                model.state_dict(),
                os.path.join(root_dir, "best_metric_model_largePatch.pth")
            )
            print(f"Model saved! Best Dice: {best_dice:.4f} at step {best_step}")
        else:
            print(f"Not saved. Best: {best_dice:.4f}, Current: {mean_dice:.4f}")
    
    if global_step >= max_iterations:
        print(f"\nReached max iterations ({max_iterations}). Stopping.")
        break

# Load best model
model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model_largePatch.pth")))
print(f"train completed, best_metric: {best_dice:.4f} at iteration: {best_step}")

# Cell 18 - Plot training curves
plt.figure("train", (12, 6))
plt.subplot(1, 2, 1)
plt.title("Iteration Average Loss")
x = [eval_num * (i + 1) for i in range(len(epoch_loss_values))]
y = epoch_loss_values
plt.xlabel("Iteration")
plt.plot(x, y)
plt.subplot(1, 2, 2)
plt.title("Val Mean Dice")
x = [eval_num * (i + 1) for i in range(len(metric_values))]
y = metric_values
plt.xlabel("Iteration")
plt.plot(x, y)
plt.savefig(os.path.join(root_dir, "training_curves.png"))
print(f"Saved training curves to {os.path.join(root_dir, 'training_curves.png')}")

print("Training complete!")
