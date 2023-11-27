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
import queue
import torchvision.transforms as transforms
from io import BytesIO
from hyperspectral.util.eval_metrics import calculate_rmse, SSIM
from skimage.metrics import structural_similarity as ssim
import warnings
import rasterio

if __name__ == '__main__':
    warnings.filterwarnings("ignore", message="Using a target size .* that is different to the input size .*")
    warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
    opt = TestOptions().parse()

    transforms_ = [transforms.Resize((opt.crop_size, opt.crop_size), Image.BICUBIC),
    # transforms.Resize(int(opt.crop_size * 1.12), Image.Resampling.BICUBIC),
    # transforms.RandomCrop(opt.crop_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    dataloader = DataLoader(MaskImageDataset(opt.datasets_dir, opt.dataroot, transforms_=transforms_, unaligned=True, mode='test'),
                batch_size=opt.batch_size, shuffle=opt.shuffle, num_workers=opt.n_cpu)
    
    model = create_model(opt)
    model.data_length = len(dataloader.dataset)
    model.setup(opt)
    rec_ssim = []
    py.mkdir(model.output_dir)
    py.mkdir(model.sample_dir)
    iters = 0

    for i, batch in tqdm.tqdm(enumerate(dataloader), desc='Test Loop', total=len(dataloader.dataset)):
        
        iters+=1
        model.set_input(batch)
        model.forward()
        real_A, fake_B = model.get_visuals(iters)
        resize_transform = transforms.Resize((opt.crop_size, opt.crop_size), Image.BICUBIC)
        real_B = Image.open(r'datasets/rgb_rmse/ref/20_rgb.png')
        real_B = resize_transform(real_B)

        ssim_orig, _ = ssim(np.mean(np.array(real_A), axis=-1), np.mean(np.array(real_B), axis=-1), full=True, data_range=255)
        ssim_artificial, _ = ssim(np.mean(np.array(fake_B), axis=-1), np.mean(np.array(real_B), axis=-1), full=True, data_range=255)
        rec_ssim.append([ssim_orig, ssim_artificial])
    
    for i, val in enumerate(rec_ssim):
        print("Reconstruction for Image: ", i+1)
        print("SSIM: GT to Original Image: ", val[0])
        print("SSIM: GT to Enhanced Image: ", val[1])
        print()