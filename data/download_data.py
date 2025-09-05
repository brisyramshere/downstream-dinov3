import os
import urllib.request
import tarfile
import tqdm

class TqdmUpTo(tqdm.tqdm):
    """Provides `update_to(block_num, block_size, total_size)`.
    From: https://github.com/tqdm/tqdm#hooks-and-callbacks
    """
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_and_extract_imagenette(root_dir='.'):
    """
    Downloads and extracts the Imagenette2 dataset.
    Imagenette2 is a smaller subset of ImageNet that is great for testing.
    """
    dataset_url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"
    dataset_name = "imagenette2"
    target_dir = os.path.join(root_dir, dataset_name)
    tar_path = os.path.join(root_dir, "imagenette2.tgz")

    if os.path.exists(target_dir):
        print(f"Dataset already exists at: {target_dir}")
        return target_dir

    print(f"Downloading Imagenette2 from {dataset_url}...")
    try:
        with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1,
                      desc=tar_path.split(os.sep)[-1]) as t:
            urllib.request.urlretrieve(dataset_url, filename=tar_path, reporthook=t.update_to)
        print("\nDownload complete.")

        print(f"Extracting {tar_path}...")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path=root_dir)
        print(f"Extraction complete. Dataset is at: {target_dir}")

        # Clean up the downloaded tar file
        os.remove(tar_path)
        print(f"Removed temporary file: {tar_path}")

    except Exception as e:
        print(f"An error occurred: {e}")
        if os.path.exists(tar_path):
            os.remove(tar_path)
        return None
    
    return target_dir

if __name__ == '__main__':
    # The script will download the data into the `downstrean-dinov3/data` directory
    data_root = os.path.dirname(os.path.abspath(__file__))
    download_and_extract_imagenette(root_dir=data_root)
