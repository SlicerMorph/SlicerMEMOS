import os
import time
import fire

# import MONAI and dependencies
import nibabel as nib
import numpy as np
import torch

from monai.config import print_config
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNETR

from monai.transforms import (
  Compose,
  EnsureChannelFirstd,
  LoadImaged,
  Orientationd,
  ScaleIntensityRanged,
  EnsureTyped,
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
      EnsureTyped(keys=["image"]),
  ])

  # set up devices
  print_config()
  torch.set_num_threads(24)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  image_dim = 128

  # UNETR architecture - updated for MONAI 1.4.0
  net = UNETR(
      in_channels=1,
      out_channels=51,
      img_size=(image_dim, image_dim, image_dim),
      feature_size=16,
      hidden_size=768,
      mlp_dim=3072,
      num_heads=12,
      proj_type="perceptron",  # Updated from pos_embed for MONAI 1.4.0
      norm_name="instance",
      conv_block=True,         # Added to match training architecture
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
    # Wrap volume_path in dictionary for MONAI transforms
    image = pre_transforms({"image": volume_path})['image']
    # Add batch dimension: [C, H, W, D] -> [1, C, H, W, D]
    image = image.unsqueeze(0).to(device)
    
    output_raw = sliding_window_inference(image, (image_dim, image_dim, image_dim), 4, net, overlap=0.8)
    output_final= torch.argmax(output_raw, dim=1).detach().cpu()[0, :, :, :]
    end = time.time()
    print("MEMOS Inference time: ", end - start)
    print("Writing: ", output_path)
    
    # Updated for MONAI 1.4.0 - use nibabel instead of deprecated write_nifti
    output_array = output_final.numpy()
    original_img = nib.load(volume_path)
    output_img = nib.Nifti1Image(output_array.astype(np.int16), original_img.affine, original_img.header)
    nib.save(output_img, output_path)

if __name__ == '__main__':
  torch.cuda.set_device(0)
  fire.Fire(main)
