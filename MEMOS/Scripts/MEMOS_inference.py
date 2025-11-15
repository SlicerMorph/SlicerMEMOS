import os
import logging
import sys
import tempfile
import shutil
import time
import fire
from glob import glob
from packaging import version

# import MONAI and dependencies
import monai
import nibabel as nib
import numpy as np
import torch
import einops

from monai.config import print_config
from monai.data import Dataset, DataLoader, create_test_image_3d, decollate_batch
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNETR

from monai.transforms import (
  Activationsd,
  AsDiscreted,
  Compose,
  EnsureChannelFirstd,
  Invertd,
  LoadImaged,
  Orientationd,
  SaveImaged,
  Spacingd,
  CropForegroundd,
  EnsureTyped,
  ScaleIntensityRanged,
)

def main(volume_path, model_path, output_path, color_node):
  # Create input dictionary for MONAI transforms
  input_dict = {"image": volume_path}
  
  # define pre-transforms
  pre_transforms = Compose([
      LoadImaged(keys=["image"]),
      EnsureChannelFirstd(keys=["image"]),
      Orientationd(keys="image", axcodes="RAS"),
      ScaleIntensityRanged(
          keys=["image"], a_min=-175, a_max=250,
          b_min=0.0, b_max=1.0, clip=True),
      EnsureTyped(keys=["image"]),
  ])

  # set up devices
  print_config()
  torch.set_num_threads(24)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  image_dim = 128

  net = UNETR(
      in_channels=1,
      out_channels=51,
      img_size=(image_dim, image_dim, image_dim),
      feature_size=16,
      hidden_size=768,
      mlp_dim=3072,
      num_heads=12,
      pos_embed="perceptron",
      norm_name="instance",
      res_block=True,
      dropout_rate=0.0,
  ).to(device)

  if device.type == "cpu":
      net.load_state_dict(torch.load(model_path, map_location='cpu'))
  else:
      net.load_state_dict(torch.load(model_path))
  net.eval()

  with torch.no_grad():
    start=time.time()
    image = pre_transforms(input_dict)['image'].to(device)
    
    # MONAI transforms return [C, H, W, D], but sliding_window_inference needs [B, C, H, W, D]
    if image.ndim == 4:
      image = image.unsqueeze(0)  # Add batch dimension: [C,H,W,D] -> [B,C,H,W,D]
    
    output_raw = sliding_window_inference(image, (image_dim, image_dim, image_dim), 4, net, overlap=0.8)
    output_final= torch.argmax(output_raw, dim=1).detach().cpu()[0, :, :, :]
    end = time.time()
    print("MEMOS Inference time: ", end - start)
    # Use nibabel to write NIfTI (MONAI 1.4+ removed write_nifti)
    nifti_img = nib.Nifti1Image(output_final.astype(np.uint8), affine=np.eye(4))
    nib.save(nifti_img, output_path)

if __name__ == '__main__':
  torch.cuda.set_device(0)
  fire.Fire(main)


