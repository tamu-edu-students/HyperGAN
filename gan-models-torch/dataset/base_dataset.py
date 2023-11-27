import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class BaseImageDataset(Dataset):
    """
    This class implements the base dataset operations that will be implemented depending on the form of data being loaded in
    """
    def __init__(self) -> None:
        """Used for setting up transforms functions as well as file operations for data retrieval from memory"""
        super().__init__()
                 

    def __getitem__(self, index):
        """used for capturing item A and item B from both domains"""
        super().__getitem__()

    def __len__(self):
        """maximum length of either dataset"""
        super().__getitem__()