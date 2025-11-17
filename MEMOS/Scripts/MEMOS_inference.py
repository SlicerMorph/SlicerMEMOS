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
  
  # Load model state dict to detect resolution
  if device.type == "cpu":
      state_dict = torch.load(model_path, map_location='cpu')
  else:
      state_dict = torch.load(model_path)
  
  # Detect image dimension from model's positional embedding size
  # The vit.patch_embedding.position_embeddings has shape [1, num_patches, hidden_size]
  # For 128^3: num_patches = (128/16)^3 = 512
  # For 192^3: num_patches = (192/16)^3 = 1728
  pos_embed_key = 'vit.patch_embedding.position_embeddings'
  if pos_embed_key in state_dict:
      num_patches = state_dict[pos_embed_key].shape[1]
      if num_patches == 512:
          image_dim = 128
          print(f"Detected 128^3 model (num_patches={num_patches})")
      elif num_patches == 1728:
          image_dim = 192
          print(f"Detected 192^3 model (num_patches={num_patches})")
      else:
          print(f"Warning: Unexpected num_patches={num_patches}, defaulting to 128^3")
          image_dim = 128
  else:
      print("Warning: Could not detect model resolution, defaulting to 128^3")
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

  net.load_state_dict(state_dict)
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
