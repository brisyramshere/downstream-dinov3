import torch
import argparse
import os
import numpy as np
import SimpleITK as sitk
from PIL import Image
from tqdm import tqdm

# Import the refactored functions from inference_segmentor
from inference_segmentor import load_model, run_batch_inference_on_images

def get_args_parser():
    parser = argparse.ArgumentParser('3D NII Segmentation Inference', add_help=False)
    parser.add_argument('--checkpoint', required=True, type=str, help='Path to the trained model checkpoint.')
    parser.add_argument('--input_nii', required=True, type=str, help='Path to the input 3D .nii.gz file.')
    parser.add_argument('--output_nii', type=str, default='output_mask_3d.nii.gz', help='Path to save the output 3D mask.')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for inference (default: 4).')
    return parser

def normalize_batch_slices_to_rgb(data_batch):
    """
    Normalizes a batch of 16-bit slices to 3-channel RGB PIL Images,
    applying the same fixed preprocessing as during training.
    
    Args:
        data_batch: List of numpy arrays (2D slices)
        
    Returns:
        List of PIL Images
    """
    batch_imgs = []
    for data in data_batch:
        # Ensure data is float for calculations
        data = data.astype(np.float32)
        
        # 1. Mean-std normalization
        mean = np.mean(data)
        std = np.std(data)
        if std > 0:
            data = (data - mean) / std
        else:
            data = np.zeros_like(data)

        # 2. Clip to [-2, 2]
        data = np.clip(data, -2.0, 2.0)
        
        # 3. Scale [-2, 2] to [0, 255]
        data = ((data + 2) / 4.0) * 255.0
        data = data.astype(np.uint8)
        
        # 4. Convert to 3-channel RGB PIL Image
        img = Image.fromarray(np.stack([data] * 3, axis=-1), 'RGB')
        
        # 5. Apply fixed gamma correction (same as training preprocessing)
        # Using gamma=0.75 to enhance low-intensity pixels and improve foreground-background boundaries
        import torchvision.transforms.functional as tf
        img = tf.adjust_gamma(img, gamma=0.75)
        
        batch_imgs.append(img)
    
    return batch_imgs

def inference3d_nii(checkpoint, input_nii, output_nii, batch_size=4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Load Model and Config ---
    try:
        model, config = load_model(checkpoint, device)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return

    # --- Load 3D NII Image ---
    if not os.path.exists(input_nii):
        print(f"Error: Input NII file not found at {input_nii}")
        return
    
    input_vol = sitk.ReadImage(input_nii)
    input_array = sitk.GetArrayFromImage(input_vol) # Shape: (depth, height, width)
    
    print(f"Loaded 3D volume with shape: {input_array.shape}")
    print(f"Using batch size: {batch_size}")

    # --- Perform Batch Inference ---
    output_slices = []
    num_slices = input_array.shape[0]
    
    print("Running batch inference on 3D volume...")
    for i in tqdm(range(0, num_slices, batch_size)):
        # Get batch of slices
        end_idx = min(i + batch_size, num_slices)
        batch_slices = []
        
        for j in range(i, end_idx):
            img_slice_np = input_array[j, :, :]
            batch_slices.append(img_slice_np)
        
        # Pre-process the batch (normalize, convert to RGB PIL images)
        batch_imgs_pil = normalize_batch_slices_to_rgb(batch_slices)
        
        # Run batch inference
        pred_masks = run_batch_inference_on_images(model, batch_imgs_pil, config, device)
        output_slices.extend(pred_masks)

    # --- Assemble and Save 3D Mask ---
    # Stack all the 2D masks to form a 3D volume
    output_array = np.stack(output_slices, axis=0).astype(np.uint8)
    
    print(f"Assembled output mask with shape: {output_array.shape}")

    # Create a new SimpleITK image from the output array
    output_vol = sitk.GetImageFromArray(output_array)
    
    # Copy metadata (spacing, origin, direction) from the input volume
    output_vol.CopyInformation(input_vol)
    
    # Save the 3D mask
    sitk.WriteImage(output_vol, output_nii)
    print(f"Saved 3D mask to: {output_nii}")


if __name__ == '__main__':
    # Uncomment for command line usage:
    # parser = argparse.ArgumentParser('3D NII Segmentation Inference', parents=[get_args_parser()])
    # args = parser.parse_args()
    # checkpoint = args.checkpoint
    # input_nii = args.input_nii
    # output_nii = args.output_nii
    # batch_size = args.batch_size
    
    # Hard-coded for testing:
    checkpoint = "output/segmentation/dinov3_vits16_unet_full_mrihead/checkpoint.pth"
    input_nii = "data/MRIhead/nii/_images/TOF_HSW_HEAD_ANGIO_0000.nii.gz"
    output_nii = "data/MRIhead/nii/_test/TOF_HSW_HEAD_ANGIO_0000.nii.gz"
    batch_size = 8  # User can modify this value
    
    inference3d_nii(checkpoint, input_nii, output_nii, batch_size)