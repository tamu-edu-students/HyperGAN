import glob
import random
import os
from .base_dataset import BaseImageDataset
from PIL import Image
import torchvision.transforms as transforms

class MaskImageDataset(BaseImageDataset):
    def __init__(self, dir, dataroot, transforms_=None, unaligned=False, mode='train'):
        """Used for setting up transforms functions as well as file operations for data retrieval from memory"""
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        if mode == 'train': #capturing items from corresponding directory depending on training mode
            self.files_A = sorted(glob.glob(os.path.join(dir, dataroot, 'trainA') + '/*.*'))
            self.files_B = sorted(glob.glob(os.path.join(dir, dataroot, 'trainB') + '/*.*'))
        elif mode == 'test':
            self.files_A = sorted(glob.glob(os.path.join(dir, dataroot, 'testA') + '/*.*'))
            self.files_B = sorted(glob.glob(os.path.join(dir, dataroot, 'testB') + '/*.*'))
            
    def __getitem__(self, index):
        """used for capturing item A and item B from both domains"""
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))  #indexing to location in file directory

        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)])) #capturing random pair from B file directory if unaligned
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))  #else, picking match from directory

        return {'A': item_A, 'B': item_B} #returning both items as a dictionary

    def __len__(self):
        """maximum length of either dataset"""
        return max(len(self.files_A), len(self.files_B))