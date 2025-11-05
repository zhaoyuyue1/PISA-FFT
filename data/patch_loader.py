
import pandas as pd
import numpy as np
import os
import cv2
import torch
from torch.utils.data import Dataset
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops

class PermeabilityDataset(Dataset):
    def __init__(self, image_dir, csv_file, patch_size=56, transform=None):
        self.image_dir = image_dir
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.patch_size = patch_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, str(self.data.iloc[idx, 0]) + ".png")
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)['image'].numpy().transpose(1, 2, 0)

        ps = self.patch_size
        h, w, c = image.shape
        num_patches_h, num_patches_w = h // ps, w // ps
        patch_feats = []

        for ch in range(c):
            feats = []
            for i in range(num_patches_h):
                for j in range(num_patches_w):
                    patch = image[i * ps:(i + 1) * ps, j * ps:(j + 1) * ps, ch]
                    thresh = threshold_otsu(patch)
                    bw = (patch > thresh).astype(np.uint8)
                    poro = np.mean(bw)
                    lbl = label(bw, connectivity=2)
                    props = regionprops(lbl)
                    area_ratio = max([prop.area for prop in props]) / (ps * ps) if props else 0.0
                    feats.append([poro, area_ratio])
            patch_feats.append(np.array(feats))

        patch_feats = np.concatenate(patch_feats, axis=1)
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        patch_feats = torch.tensor(patch_feats, dtype=torch.float32)
        label_val = self.data.iloc[idx, 1]
        log_label = np.log1p(label_val)

        return image, patch_feats, torch.tensor(log_label, dtype=torch.float32)

def create_dataloaders(dataset, seed=42, batch_size=64, num_workers=4):
    from torch.utils.data import random_split, DataLoader
    total_len = len(dataset)
    train_len = int(0.6 * total_len)
    valid_len = int(0.2 * total_len)
    test_len = total_len - train_len - valid_len
    train_dataset, valid_dataset, test_dataset = random_split(
        dataset, [train_len, valid_len, test_len],
        generator=torch.Generator().manual_seed(seed)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, valid_loader, test_loader, valid_dataset, train_dataset
