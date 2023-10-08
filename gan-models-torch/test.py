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
from dataset.maskshadow_dataset import MaskImageDataset
import functools
import torchvision.transforms as transforms

if __name__ == '__main__':
    
    opt = TestOptions().parse()
    model = create_model(opt)
    model.setup(opt)


    transforms_ = [#transforms.Resize((opt.size, opt.size), Image.BICUBIC),
    transforms.Resize(int(opt.crop_size * 1.12), Image.Resampling.BICUBIC),
    transforms.RandomCrop(opt.crop_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    dataloader = DataLoader(MaskImageDataset(opt.datasets_dir, opt.dataroot, transforms_=transforms_, unaligned=True, mode='test'),
                batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)

    to_pil = transforms.ToPILImage()
    iters = 0
    py.mkdir(opt.results_dir)

    for i, batch in tqdm.tqdm(enumerate(dataloader), desc='Test Loop', total=len(dataloader.dataset)):
        model.set_input(batch)
        iters += 1
        model.forward()

        img_real_A = util.mod_to_pil(model.real_A)
        img_fake_B = util.mod_to_pil(model.fake_B)
        
        img_real_B = util.mod_to_pil(model.real_B)
        img_fake_A = util.mod_to_pil(model.fake_A)
        

        images = [img_real_A, img_fake_B, img_real_B, img_fake_A]
        num_rows = 2
        num_columns = 2
        image_width, image_height = images[0].size
        output_width = num_columns * image_width
        output_height = num_rows * image_height

        output_image = Image.new('RGB', (output_width, output_height))
        # Paste the individual PIL images onto the output image
        for i, image in enumerate(images):
            row = i // num_columns
            col = i % num_columns
            x_offset = col * image_width
            y_offset = row * image_height
            output_image.paste(image, (x_offset, y_offset))

        # Save the combined image as a PNG file
        output_image.save(py.join(opt.results_dir, 'img-{}.jpg'.format(iters)))


