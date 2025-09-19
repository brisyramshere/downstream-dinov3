import os
from PIL import Image
from torch.utils.data import Dataset
import torch

class Imagenette2Dataset(Dataset):
    """
    A PyTorch Dataset for ImageNet-style classification datasets.

    Assumes the directory structure is:
    /path/to/dataset/
    ├── train/
    │   ├── class1/
    │   │   ├── img1.jpg
    │   │   └── ...
    │   └── class2/
    │       └── ...
    └── val/
        ├── class1/
        │   └── ...
        └── class2/
            └── ...
    """
    def __init__(self, root_dir: str, split: str = 'train', transform=None):
        """
        Args:
            root_dir (str): The root directory of the dataset.
            split (str): The dataset split, e.g., 'train' or 'val'.
            transform (callable, optional): A function/transform to apply to the images.
        """
        self.root_dir = root_dir
        self.split_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.samples = []
        self.classes = sorted(os.listdir(self.split_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        if not self.classes:
            raise FileNotFoundError(f"No class folders found in {self.split_dir}")

        self._make_dataset()

    def _make_dataset(self):
        """Scans the directory and builds the list of samples."""
        for class_name in self.classes:
            class_idx = self.class_to_idx[class_name]
            class_dir = os.path.join(self.split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            
            for file_name in sorted(os.listdir(class_dir)):
                if self._is_image_file(file_name):
                    path = os.path.join(class_dir, file_name)
                    item = (path, class_idx)
                    self.samples.append(item)

    def _is_image_file(self, filename: str) -> bool:
        """Checks if a file is a common image format."""
        return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif', '.tiff'))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, target = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy image and target on error
            return torch.zeros(3, 224, 224), -1

        if self.transform:
            image = self.transform(image)
        
        return image, target

    def get_num_classes(self) -> int:
        return len(self.classes)

if __name__ == '__main__':
    # Example usage (requires a dummy dataset structure)
    # 1. Create a dummy dataset
    print("Creating a dummy dataset for testing...")
    dummy_root = './dummy_imagenet'
    os.makedirs(os.path.join(dummy_root, 'train', 'cat'), exist_ok=True)
    os.makedirs(os.path.join(dummy_root, 'train', 'dog'), exist_ok=True)
    os.makedirs(os.path.join(dummy_root, 'val', 'cat'), exist_ok=True)
    os.makedirs(os.path.join(dummy_root, 'val', 'dog'), exist_ok=True)

    try:
        Image.new('RGB', (100, 100), color = 'red').save(os.path.join(dummy_root, 'train', 'cat', 'cat1.jpg'))
        Image.new('RGB', (100, 100), color = 'blue').save(os.path.join(dummy_root, 'train', 'dog', 'dog1.jpg'))
        Image.new('RGB', (100, 100), color = 'green').save(os.path.join(dummy_root, 'val', 'cat', 'cat_val1.jpg'))
        Image.new('RGB', (100, 100), color = 'yellow').save(os.path.join(dummy_root, 'val', 'dog', 'dog_val1.jpg'))
        print("Dummy dataset created.")

        # 2. Test the dataset class
        print("\nTesting Imagenette2Dataset...")
        train_dataset = Imagenette2Dataset(root_dir=dummy_root, split='train')
        val_dataset = Imagenette2Dataset(root_dir=dummy_root, split='val')

        print(f"Number of training samples: {len(train_dataset)}")
        print(f"Number of validation samples: {len(val_dataset)}")
        print(f"Number of classes: {train_dataset.get_num_classes()}")
        print(f"Classes: {train_dataset.classes}")
        print(f"Class to index mapping: {train_dataset.class_to_idx}")

        # 3. Test getting an item
        img, label = train_dataset[0]
        print(f"\nSample 0: Image type: {type(img)}, Label: {label} (Class: {train_dataset.classes[label]})")
        
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        train_dataset.transform = transform
        img_tensor, label = train_dataset[0]
        print(f"Transformed Sample 0: Image shape: {img_tensor.shape}, Label: {label}")

    finally:
        # 4. Clean up the dummy dataset
        import shutil
        if os.path.exists(dummy_root):
            shutil.rmtree(dummy_root)
            print(f"\nCleaned up dummy dataset at {dummy_root}")
