import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class BaseImageDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
                 

    def __getitem__(self, index):
        super().__getitem__()

    def __len__(self):
        super().__getitem__()