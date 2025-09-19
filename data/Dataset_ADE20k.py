import collections
import torch
import torchvision
import numpy as np
import glob
from PIL import Image
import torchvision.transforms as transforms
from data.dinov3_transforms import make_transform_LVD_1689M
from torch.utils import data

class ADE20Dataset(data.Dataset):
    def __init__(
        self,
        root,
        input_size,
        split="training",
        augmentations=None
    ):
        self.root = root
        self.split = split
        self.augmentations = augmentations
        self.n_classes = 150
        self.files = collections.defaultdict(list)
        self.input_size = input_size
        # Use glob for recursive file search
        self.files[self.split] = glob.glob(
            self.root + f"/images/{self.split}/**/*.jpg", recursive=True
        )
        
        self.transform_resize_and_norm = make_transform_LVD_1689M(input_size)

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_path = self.files[self.split][index].rstrip()
        # Correctly form the annotation path
        lbl_path = img_path.replace("/images/", "/annotations/").replace(".jpg", ".png")

        img = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)
        lbl = np.array(Image.open(lbl_path).convert('L'), dtype=np.uint8)

        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)

        img = self.transform_resize_and_norm(img)

        lbl = Image.fromarray(lbl.astype(np.uint8)).resize((self.img_size[0], self.img_size[1]), Image.NEAREST)
        lbl = torch.from_numpy(np.array(lbl)).long()
        # Remap labels: 0 -> 255 (ignore_index), 1-150 -> 0-149
        lbl[lbl == 0] = 255
        lbl = lbl - 1
        lbl[lbl == 254] = 255

        return img, lbl

    
   
