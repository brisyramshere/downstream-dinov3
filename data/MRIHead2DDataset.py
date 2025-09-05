import collections
import torch
import torchvision
import numpy as np
import glob
from PIL import Image
import torchvision.transforms as transforms

from torch.utils import data

class MRIHead2DDataset(data.Dataset):
    def __init__(
        self,
        root,
        split="training",
        is_transform=True,
        img_size=512,
        augmentations=None,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ):
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.n_classes = 2
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.mean = mean
        self.std = std
        self.files = collections.defaultdict(list)

        # Use glob for recursive file search for .png files
        self.files[self.split] = glob.glob(
            self.root + f"/images/{self.split}/**/*.png", recursive=True
        )
        
        self.transform_to_tensor_and_normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_path = self.files[self.split][index].rstrip()
        # Correctly form the annotation path for our new structure
        lbl_path = img_path.replace("/images/", "/annotations/")

        try:
            img = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)
            lbl = np.array(Image.open(lbl_path).convert('L'), dtype=np.uint8)
        except FileNotFoundError:
            print(f"Could not find file: {img_path} or {lbl_path}")
            # Return a dummy sample or raise an error
            return self.__getitem__((index + 1) % len(self))


        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        return img, lbl

    def transform(self, img, lbl):
        # Resize image and label
        img = Image.fromarray(img).resize((self.img_size[0], self.img_size[1]), Image.BILINEAR)
        lbl = Image.fromarray(lbl.astype(np.uint8)).resize((self.img_size[0], self.img_size[1]), Image.NEAREST)
        lbl = torch.from_numpy(np.array(lbl)).long()

        # Unlike ADE20k, we don't need to remap the labels for this dataset.
        # We assume the label values are already correct (e.g., 0 for background, 1, 2, ... for classes)
        # The ignore_index will be handled by the loss function directly if needed.

        # Apply ToTensor and normalization to the image
        img = self.transform_to_tensor_and_normalize(img)

        return img, lbl