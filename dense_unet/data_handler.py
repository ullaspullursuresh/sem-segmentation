import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


class NanoSEMDataset(Dataset):
    def __init__(self, image_dir, mask_dir, size=(256, 256), augment=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.size = size
        self.augment = augment

        # Sorted for reproducibility; keep only images that have a matching mask
        self.images = []
        for f in sorted(os.listdir(image_dir)):
            if not f.lower().endswith(".png"):
                continue
            base = os.path.splitext(f)[0]
            if os.path.exists(os.path.join(mask_dir, f"{base}_mask.png")):
                self.images.append(f)

        if not self.images:
            raise RuntimeError("No image/mask pairs found. Check paths and naming.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        base = os.path.splitext(img_name)[0]

        # Context managers close the file handles promptly
        with Image.open(os.path.join(self.image_dir, img_name)) as im:
            image = im.convert("L")
        with Image.open(os.path.join(self.mask_dir, f"{base}_mask.png")) as m:
            mask = m.convert("L")

        # Resize: bilinear for the image, nearest for the mask (keeps labels crisp)
        image = TF.resize(image, self.size, interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.size, interpolation=InterpolationMode.NEAREST)

        # Paired augmentation: same random decision applied to both
        if self.augment:
            if random.random() < 0.5:
                image, mask = TF.hflip(image), TF.hflip(mask)
            if random.random() < 0.5:
                image, mask = TF.vflip(image), TF.vflip(mask)
            if random.random() < 0.5:
                angle = random.choice([90, 180, 270])
                image, mask = TF.rotate(image, angle), TF.rotate(mask, angle)

        # To tensors, shape (1, H, W)
        image = TF.to_tensor(image)                 # float in [0, 1]
        image = TF.normalize(image, [0.5], [0.5])   # normalize image only, never the mask
        mask = (TF.to_tensor(mask) > 0).float()     # binary 0/1

        return image, mask