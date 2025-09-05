import os
import urllib.request
import zipfile
import tqdm

class TqdmUpTo(tqdm.tqdm):
    """Provides `update_to(block_num, block_size, total_size)`.
    From: https://github.com/tqdm/tqdm#hooks-and-callbacks
    """
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_and_extract_ade20k(root_dir='.'):
    """
    Downloads and extracts the ADE20K dataset from a public mirror.
    
    NOTE: The official dataset requires registration. This uses a non-official mirror
    for convenience. If this link fails, you may need to download it manually.
    """
    # URL for the dataset zip file
    dataset_url = "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"
    dataset_name = "ade20k"
    target_dir = os.path.join(root_dir, dataset_name)
    zip_path = os.path.join(root_dir, "ADEChallengeData2016.zip")

    if os.path.exists(target_dir):
        print(f"Dataset directory already exists at: {target_dir}")
        print("Skipping download and extraction.")
        return target_dir

    print(f"Downloading ADE20K from {dataset_url}...")
    print("This is a large file (~1.8GB) and may take a while.")

    try:
        with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1,
                      desc=zip_path.split(os.sep)[-1]) as t:
            urllib.request.urlretrieve(dataset_url, filename=zip_path, reporthook=t.update_to)
        print("\nDownload complete.")

        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for member in tqdm.tqdm(zip_ref.infolist(), desc='Extracting '):
                zip_ref.extract(member, root_dir)
        print("Extraction complete.")
        
        # The extracted folder is named ADEChallengeData2016, we rename it to ade20k
        extracted_folder_path = os.path.join(root_dir, 'ADEChallengeData2016')
        if os.path.exists(extracted_folder_path):
            os.rename(extracted_folder_path, target_dir)
            print(f"Renamed extracted folder to: {target_dir}")
        else:
            print(f"Warning: Expected extracted folder not found at {extracted_folder_path}")

    except Exception as e:
        print(f"An error occurred during download or extraction: {e}")
        return None
    finally:
        # Clean up the downloaded zip file
        if os.path.exists(zip_path):
            os.remove(zip_path)
            print(f"Removed temporary file: {zip_path}")
    
    return target_dir

if __name__ == '__main__':
    # The script will download the data into the `downstrean-dinov3/data` directory
    data_root = os.path.dirname(os.path.abspath(__file__))
    download_and_extract_ade20k(root_dir=data_root)
