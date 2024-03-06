import time
from options.test_options import TestOptions
from util import util
from models import networks, create_model
from models import create_model
import pylib as py
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from PIL import Image
import tqdm
from dataset.maskshadow_dataset import MaskImageDataset
<<<<<<< HEAD
from dataset.hyperspectral_dataset import HyperspectralImageDataset
=======
>>>>>>> 2f4463fff66c40eb98ff6fa17c80c0caee775ac4
import queue
import torchvision.transforms as transforms
from io import BytesIO
import warnings
import rasterio

if __name__ == "__main__":
    warnings.filterwarnings(
        "ignore",
        message="Using a target size .* that is different to the input size .*",
    )
    warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
    opt = TestOptions().parse()

<<<<<<< HEAD
    # transforms_ = [
    #     transforms.Resize((opt.crop_size, opt.crop_size), Image.BICUBIC),
    #     # transforms.Resize(int(opt.crop_size * 1.12), Image.Resampling.BICUBIC),
    #     # transforms.RandomCrop(opt.crop_size),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # ]
    # dataloader = DataLoader(
    #     MaskImageDataset(
    #         opt.datasets_dir,
    #         opt.dataroot,
    #         transforms_=transforms_,
    #         unaligned=True,
    #         mode="test",
    #     ),
    #     batch_size=opt.batch_size,
    #     shuffle=opt.shuffle,
    #     num_workers=opt.n_cpu,
    # )

    mean_values = np.array([0.5] * opt.input_nc)  # input_nc channels with a mean of 0.5
    std_values = np.array([0.5] * opt.input_nc)   # input_nc channels with a standard deviation of 0.5

    transforms_ = [
    transforms.ToTensor(),
    transforms.Normalize(mean_values.tolist(), std_values.tolist())
    ]
    dataloader = DataLoader(HyperspectralImageDataset(opt.datasets_dir, opt.dataroot, True, transforms_=transforms_, unaligned=True, mode='test'),
                batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)
=======
    transforms_ = [
        transforms.Resize((opt.crop_size, opt.crop_size), Image.BICUBIC),
        # transforms.Resize(int(opt.crop_size * 1.12), Image.Resampling.BICUBIC),
        # transforms.RandomCrop(opt.crop_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    dataloader = DataLoader(
        MaskImageDataset(
            opt.datasets_dir,
            opt.dataroot,
            transforms_=transforms_,
            unaligned=True,
            mode="test",
        ),
        batch_size=opt.batch_size,
        shuffle=opt.shuffle,
        num_workers=opt.n_cpu,
    )
>>>>>>> 2f4463fff66c40eb98ff6fa17c80c0caee775ac4

    model = create_model(opt)
    model.data_length = len(dataloader.dataset)
    model.setup(opt)

    to_pil = transforms.ToPILImage()
    iters = 0
    py.mkdir(model.output_dir)
    py.mkdir(model.sample_dir)

    for i, batch in tqdm.tqdm(
        enumerate(dataloader), desc="Test Loop", total=len(dataloader.dataset)
    ):

        model.set_input(batch)
        iters += 1

        model.forward()
        model.get_visuals(iters)

        # if iters == 200:
        #     break
    model.expand_dataset() if opt.expand_dataset else None
