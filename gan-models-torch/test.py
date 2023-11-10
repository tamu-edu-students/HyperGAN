import time
from options.test_options import TestOptions 
from util import util
from models import networks, create_model
from models import create_model
import pylib as py
import numpy as np
import imlib as im
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from PIL import Image
import tqdm
import spectral
from dataset.maskshadow_dataset import MaskImageDataset
import functools
import torchvision.transforms as transforms

if __name__ == '__main__':
    
    opt = TestOptions().parse()

    transforms_ = [#transforms.Resize((opt.size, opt.size), Image.BICUBIC),
    transforms.Resize(int(opt.crop_size * 1.12), Image.Resampling.BICUBIC),
    transforms.RandomCrop(opt.crop_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    dataloader = DataLoader(MaskImageDataset(opt.datasets_dir, opt.dataroot, transforms_=transforms_, unaligned=True, mode='test'),
                batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)
    
    model = create_model(opt)
    model.data_length = len(dataloader.dataset)
    model.setup(opt)

    
    to_pil = transforms.ToPILImage()
    iters = 0
    py.mkdir(model.output_dir)
    py.mkdir(model.sample_dir)

    for i, batch in tqdm.tqdm(enumerate(dataloader), desc='Test Loop', total=len(dataloader.dataset)):
        
        model.set_input(batch)
        iters += 1
        
        model.forward()
        model.get_visuals(iters)

        if iters == 200:
            break
    #model.expand_dataset()



