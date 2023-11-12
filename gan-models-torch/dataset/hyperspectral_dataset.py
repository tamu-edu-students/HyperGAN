import glob
import random
import os
import torch
from hyperspectral import processor
from .base_dataset import BaseImageDataset
from PIL import Image
import torchvision.transforms as transforms

class HyperspectralImageDataset(BaseImageDataset):
    def __init__(self, dir, dataroot, loadRGB, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.loadRGB = loadRGB

        if mode == 'train':
            self.files_A = sorted(glob.glob(os.path.join(dir, dataroot, 'trainA') + '/*.tiff*'))
            self.files_B = sorted(glob.glob(os.path.join(dir, dataroot, 'trainB') + '/*.tiff*'))
        elif mode == 'test':
            self.files_A = sorted(glob.glob(os.path.join(dir, dataroot, 'testA') + '/*.tiff*'))
            self.files_B = sorted(glob.glob(os.path.join(dir, dataroot, 'testB') + '/*.tiff*'))
            
    def __getitem__(self, index):
        item_A = self.files_A[index % len(self.files_A)]
        proc = processor.Processor()
        item_A = self.transform(proc.prepare_data(item_A))
        #item_A = item_A / np.max(item_A)
        #item_A = torch.from_numpy(item_A)
        #item_A = torch.squeeze(item_A)

        if self.unaligned:          
            item_B = self.files_B[random.randint(0, len(self.files_B) - 1)]
        else:
            item_B = self.files_B[index % len(self.files_B)]

        item_B = self.transform(proc.prepare_data(item_B))
        #item_B = item_B / np.max(item_B)
        #item_B = torch.squeeze(item_B)

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))