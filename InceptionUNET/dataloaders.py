import glob
import os

import numpy as np
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

from .augmentations import RandomFlip, toTensor
from .config import *


class PrimaLayout(Dataset):
    def __init__(self, root, img_folder, masks_folder, transforms=None):
        self.paths = glob.glob(root + masks_folder + '*.npy')
        self.img_paths = []
        self.npy_paths = []
        for item in self.paths:
            img_path = item.replace(masks_folder, img_folder)
            if os.path.exists(img_path):
                self.img_paths.append(img_path)
                self.npy_paths.append(item)

        self.transform = transforms

    def __len__(self):
        return len(self.npy_paths)

    def __getitem__(self, idx):
        target = np.load(self.npy_paths[idx])
        img = np.load(self.img_paths[idx])
        filename = self.img_paths[idx].split('/')[-1]
        non_normalized = img[:, :, :3]

        sample = {"image": img, "masks": target, "filename": filename}
        if self.transform:
            sample = self.transform(sample)

        sample["no_norm"] = non_normalized
        return sample


train_transform = T.Compose([
    RandomFlip(),
    toTensor()])

val_transform = T.Compose([
    toTensor(val=True)])

train_dataset = PrimaLayout('.', TRAIN_IMAGE_FOLDER, TRAIN_MASKS_FOLDER, train_transform)
validation_dataset = PrimaLayout('.', VAL_IMAGE_FOLDER, VAL_MASKS_FOLDER, val_transform)

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=6)
validation_loader = DataLoader(validation_dataset, shuffle=False, batch_size=BATCH_SIZE, num_workers=6)
