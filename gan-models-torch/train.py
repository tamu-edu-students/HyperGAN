import time
from options.train_options import TrainOptions 
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
from dataset.hyperspectral_dataset import HyperspectralImageDataset
import functools
import torchvision.transforms as transforms

if __name__ == '__main__':
    
    opt = TrainOptions().parse()

    # transforms_ = [#transforms.Resize((opt.size, opt.size), Image.BICUBIC),
    #     transforms.Resize(int(opt.crop_size * 1.12), Image.Resampling.BICUBIC),
    #     transforms.RandomCrop(opt.crop_size),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    mean_values = np.array([0.5] * opt.input_nc)  # 51 channels with a mean of 0.5
    std_values = np.array([0.5] * opt.input_nc)   # 51 channels with a standard deviation of 0.5

    transforms_ = [
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean_values.tolist(), std_values.tolist())
        ]
    
    # dataloader = DataLoader(MaskImageDataset(opt.datasets_dir, opt.dataroot, transforms_=transforms_, unaligned=True, mode='train'),
    #             batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)
    dataloader = DataLoader(HyperspectralImageDataset(opt.datasets_dir, opt.dataroot, True, transforms_=transforms_, unaligned=True, mode='train'),
                batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)
   
    model = create_model(opt)
    model.data_length = len(dataloader.dataset)
    model.setup(opt)
    

    plt.ioff()
    curr_iter = 0
    G_losses = []
    D_A_losses = []
    D_B_losses = []
    to_pil = transforms.ToPILImage()
    total_iters = 0

    
    
    py.mkdir(model.output_dir)
    py.mkdir(model.sample_dir)
    py.mkdir(model.checkpoint_dir)


    for epoch in tqdm.trange(opt.epoch_count, opt.epochs, desc='Epoch Loop'):

        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0
        losses_temp = None

        for i, batch in tqdm.tqdm(enumerate(dataloader), desc='Inner Epoch Loop', total=len(dataloader.dataset)):
            
            model.set_input(batch)
            iter_start_time = time.time()
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            model.optimize_parameters()
            losses_temp = model.get_current_losses()

            if total_iters % 15 == 0:
                model.get_visuals(epoch_iter, epoch)
               
            iter_data_time = time.time()

        model.update_learning_rate()

        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks(epoch, losses_temp, opt)
            
