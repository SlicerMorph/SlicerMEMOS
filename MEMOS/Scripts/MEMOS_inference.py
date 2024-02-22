import logging
import os
import sys
import tempfile
import shutil
import time
import fire
from glob import glob
from packaging import version

# import MONAI and dependencies
import nibabel as nib
import numpy as np
import torch
import einops

from monai.config import print_config
from monai.data import Dataset, DataLoader, create_test_image_3d, decollate_batch
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNETR
from monai.data.nifti_writer import write_nifti

from monai.transforms import (
  Activationsd,
  AsDiscreted,
  AddChanneld,
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
  ToTensord
)

def main(volume_path, model_path, output_path, color_node):
  # define pre-transforms
  pre_transforms = Compose([
      LoadImaged(keys=["image"]),
      EnsureChannelFirstd(keys=["image"]),
      Orientationd(keys="image", axcodes="RAS"),
      ScaleIntensityRanged(
          keys=["image"], a_min=-175, a_max=250,
          b_min=0.0, b_max=1.0, clip=True),
      AddChanneld(keys=["image"]),
      ToTensord(keys=["image"]),
  ])

  # set up model
  os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

  # set GPU
  if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    print("Using device: ", os.environ["CUDA_VISIBLE_DEVICES"])
  # check configuration
  print_config()
  torch.set_num_threads(24)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print("Using device: ", device)
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
    image = pre_transforms(volume_path)['image'].to(device)
    output_raw = sliding_window_inference(image, (image_dim, image_dim, image_dim), 4, net, overlap=0.8)
    output_final= torch.argmax(output_raw, dim=1).detach().cpu()[0, :, :, :]
    end = time.time()
    print("MEMOS Inference time: ", end - start)
    print("Writing: ", output_path)
    write_nifti(
      data=output_final,
      file_name=output_path
      )

if __name__ == '__main__':
  fire.Fire(main)



