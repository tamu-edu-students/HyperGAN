import glob
import random
import os

from .base_dataset import BaseImageDataset
from PIL import Image
import torchvision.transforms as transforms

class MaskImageDataset(BaseImageDataset):
    def __init__(self, dir, dataroot, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        if mode == 'train':
            self.files_A = sorted(glob.glob(os.path.join(dir, dataroot, 'trainA') + '/*.*'))
            self.files_B = sorted(glob.glob(os.path.join(dir, dataroot, 'trainB') + '/*.*'))
        elif mode == 'test':
            self.files_A = sorted(glob.glob(os.path.join(dir, dataroot, 'testA') + '/*.*'))
            self.files_B = sorted(glob.glob(os.path.join(dir, dataroot, 'testB') + '/*.*'))
            
    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))

        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))