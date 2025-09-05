import os
import glob
import numpy as np
import SimpleITK as sitk
from PIL import Image
from tqdm import tqdm
import shutil

def normalize_to_uint8(data):
    """
    Normalize the 16-bit image data to 8-bit (0-255) as per user specification.
    1. Mean-std normalization.
    2. Clip to [-2, 2].
    3. Scale [-2, 2] to [0, 255].
    """
    # Ensure data is float for calculations
    data = data.astype(np.float32)
    
    # 1. Mean-std normalization
    mean = np.mean(data)
    std = np.std(data)
    if std > 0:
        data = (data - mean) / std
    else:
        # If std is zero, the image is constant. We can just make it black.
        return np.zeros_like(data, dtype=np.uint8)

    # 2. Clip to [-2, 2]
    data = np.clip(data, -2.0, 2.0)
    
    # 3. Scale [-2, 2] to [0, 255]
    # Formula: new_value = ((old_value - old_min) / (old_max - old_min)) * new_range + new_min
    # Here: ((data - (-2)) / (2 - (-2))) * 255 = ((data + 2) / 4) * 255
    data = ((data + 2) / 4.0) * 255.0
    
    return data.astype(np.uint8)

def process_nii_to_2d(
    image_path, 
    label_path, 
    output_image_dir, 
    output_label_dir, 
    slice_interval=5
):
    """
    Processes a single pair of 3D .nii.gz files, extracts 2D slices,
    normalizes them, and saves them as PNG files.
    """
    try:
        # Load image and label volumes
        img_vol = sitk.ReadImage(image_path)
        lbl_vol = sitk.ReadImage(label_path)

        img_array = sitk.GetArrayFromImage(img_vol)  # Shape: (depth, height, width)
        lbl_array = sitk.GetArrayFromImage(lbl_vol)

        # Get the base filename to create unique names for each slice
        base_filename = os.path.basename(image_path).replace('.nii.gz', '')

        # Iterate through slices with the specified interval
        for i in range(0, img_array.shape[0], slice_interval):
            img_slice = img_array[i, :, :]
            lbl_slice = lbl_array[i, :, :]

            # Skip slices that are mostly empty (optional, but good practice)
            if np.sum(img_slice) == 0:
                continue

            # --- Process Image Slice ---
            # 1. Normalize to 0-255 uint8
            img_slice_normalized = normalize_to_uint8(img_slice)
            # 2. Convert to 3-channel RGB
            img_slice_rgb = np.stack([img_slice_normalized] * 3, axis=-1)
            
            # --- Save Image and Label ---
            img_pil = Image.fromarray(img_slice_rgb, 'RGB')
            lbl_pil = Image.fromarray(lbl_slice.astype(np.uint8), 'L') # Labels are single channel

            slice_filename = f"{base_filename}_slice_{i:04d}.png"
            img_pil.save(os.path.join(output_image_dir, slice_filename))
            lbl_pil.save(os.path.join(output_label_dir, slice_filename))

    except Exception as e:
        print(f"Error processing {image_path}: {e}")


def main():
    # --- Configuration ---
    input_dir = "data/MRIhead/nii"
    output_dir = "data/MRIHead2D"
    train_split_ratio = 0.8
    slice_interval = 5 # As requested: 5-select-1

    # --- Setup Paths ---
    image_files = sorted(glob.glob(os.path.join(input_dir, "_images", "*.nii.gz")))
    label_files = sorted(glob.glob(os.path.join(input_dir, "_labels", "*.nii.gz")))

    # Create output directories
    train_img_dir = os.path.join(output_dir, "images", "training")
    train_lbl_dir = os.path.join(output_dir, "annotations", "training")
    val_img_dir = os.path.join(output_dir, "images", "validation")
    val_lbl_dir = os.path.join(output_dir, "annotations", "validation")

    for path in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        os.makedirs(path, exist_ok=True)

    # --- Data Splitting & Matching ---
    # Find the intersection of image and label files to ensure pairs exist
    image_basenames = {os.path.basename(p) for p in image_files}
    label_basenames = {os.path.basename(p) for p in label_files}
    
    matched_basenames = sorted(list(image_basenames.intersection(label_basenames)))
    
    if len(matched_basenames) < len(image_basenames):
        print("Warning: Some image files do not have corresponding label files and will be skipped.")
        unmatched_images = image_basenames - label_basenames
        print(f"Unmatched images: {unmatched_images}")

    # Create full paths for the matched files
    matched_image_files = [os.path.join(input_dir, "_images", bn) for bn in matched_basenames]

    # Simple split based on the matched file list
    num_files = len(matched_image_files)
    split_index = int(num_files * train_split_ratio)
    
    train_files = matched_image_files[:split_index]
    val_files = matched_image_files[split_index:]

    print(f"Found {len(image_files)} image volumes and {len(label_files)} label volumes.")
    print(f"Processing {num_files} matched pairs.")
    print(f"Splitting into {len(train_files)} training volumes and {len(val_files)} validation volumes.")

    # --- Processing ---
    # Process training files
    print("\nProcessing training data...")
    for img_path in tqdm(train_files):
        lbl_path = img_path.replace("_images", "_labels")
        # We already ensured the label exists from the intersection logic
        process_nii_to_2d(img_path, lbl_path, train_img_dir, train_lbl_dir, slice_interval)

    # Process validation files
    print("\nProcessing validation data...")
    for img_path in tqdm(val_files):
        lbl_path = img_path.replace("_images", "_labels")
        # We already ensured the label exists from the intersection logic
        process_nii_to_2d(img_path, lbl_path, val_img_dir, val_lbl_dir, slice_interval)

    print("\nData processing complete.")
    print(f"2D data saved in: {output_dir}")


if __name__ == "__main__":
    main()