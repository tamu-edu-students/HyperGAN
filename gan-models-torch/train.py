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
from dataset.maskshadow_dataset import MaskImageDataset 
import functools
import torchvision.transforms as transforms

if __name__ == '__main__':
    
    opt = TrainOptions().parse()

    transforms_ = [#transforms.Resize((opt.size, opt.size), Image.BICUBIC),
        transforms.Resize(int(opt.crop_size * 1.12), Image.Resampling.BICUBIC),
        transforms.RandomCrop(opt.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    # mean_values = np.array([0.5] * opt.input_nc)  # input_nc channels with a mean of 0.5
    # std_values = np.array([0.5] * opt.input_nc)   # input_nc channels with a standard deviation of 0.5

    # transforms_ = [
    #     #transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean_values.tolist(), std_values.tolist())
    #     ]
    
    dataloader = DataLoader(MaskImageDataset(opt.datasets_dir, opt.dataroot, transforms_=transforms_, unaligned=True, mode='train'),
                batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)
 #   dataloader = DataLoader(HyperspectralImageDataset(opt.datasets_dir, opt.dataroot, True, transforms_=transforms_, unaligned=True, mode='train'),
    #             batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)
   
    
    model = create_model(opt)

    model.data_length = len(dataloader.dataset)
    model.setup(opt)
    

    G_losses, D_A_losses, D_B_losses = model.G_losses, model.D_A_losses, model.D_B_losses
    plt.ioff()
    curr_iter = 0
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
        G_loss_temp = 0
        D_A_loss_temp = 0
        D_B_loss_temp = 0

        for i, batch in tqdm.tqdm(enumerate(dataloader), desc='Inner Epoch Loop', total=len(dataloader.dataset)):
            
            model.set_input(batch)
            iter_start_time = time.time()
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            model.optimize_parameters()
            losses_temp = model.get_current_losses()
            G_loss_temp += losses_temp["G_A2B"]
            D_A_loss_temp += losses_temp["D_A"]
            D_B_loss_temp += losses_temp["D_B"]

            if total_iters % 100 == 0:
                model.get_visuals(epoch_iter, epoch)
                # model.gen_rec_success()
               
            iter_data_time = time.time()
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

        model.update_learning_rate()

        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            G_losses.append(G_loss_temp / epoch_iter)
            D_A_losses.append(D_A_loss_temp / epoch_iter)
            D_B_losses.append(D_B_loss_temp/ epoch_iter)
            model.save_networks(epoch, G_losses, D_A_losses, D_B_losses, opt)
            model.plot_losses(epoch, opt.save_epoch_freq, G_losses, D_A_losses, D_B_losses)
            #model.gen_rec_success()
            
            
