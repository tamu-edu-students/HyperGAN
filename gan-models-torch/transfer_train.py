import time
from options.train_options import TrainOptions
from util import util
from models import networks, create_model
from models import create_model
import pylib as py
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from PIL import Image
import tqdm
from dataset.hyperspectral_dataset import HyperspectralImageDataset
from dataset.maskshadow_dataset import MaskImageDataset
import functools
import torchvision.transforms as transforms
import warnings
import torch
import rasterio

if __name__ == "__main__":
    warnings.filterwarnings(
        "ignore",
        message="Using a target size .* that is different to the input size .*",
    )
    warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

    opt = TrainOptions().parse()

    if opt.input_nc == 3:
        transforms_ = [  # transforms.Resize((opt.size, opt.size), Image.BICUBIC),
            transforms.Resize(int(opt.crop_size * 1.12), Image.Resampling.BICUBIC),
            transforms.RandomCrop(opt.crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
        dataloader = DataLoader(
            MaskImageDataset(
                opt.datasets_dir,
                opt.dataroot,
                transforms_=transforms_,
                unaligned=True,
                mode="train",
            ),
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.n_cpu,
        )
    else:
        mean_values = np.array(
            [0.5] * opt.input_nc
        )  # input_nc channels with a mean of 0.5
        std_values = np.array(
            [0.5] * opt.input_nc
        )  # input_nc channels with a standard deviation of 0.5

        transforms_ = [
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean_values.tolist(), std_values.tolist()),
        ]

        dataloader = DataLoader(
            HyperspectralImageDataset(
                opt.datasets_dir,
                opt.dataroot,
                True,
                transforms_=transforms_,
                unaligned=True,
                mode="train",
            ),
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.n_cpu,
        )

    model = create_model(opt)

    model.data_length = len(dataloader.dataset)
    model.setup(opt)

    G_layers_to_unfreeze = []
    D_layers_to_unfreeze = []

    if opt.unfreeze_layers_iters > 0:
        for i in range(1, opt.unfreeze_layers_iters + 1):
            G_layer_count = input(
                "For round {} of unfreezing during training, pick how many Generator layers you would like to unfreeze: ".format(
                    i
                )
            )
            D_layer_count = input(
                "For round {} of unfreezing during training, pick how many Discriminator layers you would like to unfreeze: ".format(
                    i
                )
            )
            G_layers_to_unfreeze.append(int(G_layer_count))
            D_layers_to_unfreeze.append(int(D_layer_count))

        model.gradually_unfreeze_layers(
            opt, G_layers_to_unfreeze[0], D_layers_to_unfreeze[0], 0
        )

    G_losses, D_A_losses, D_B_losses = (
        model.G_losses,
        model.D_A_losses,
        model.D_B_losses,
    )
    plt.ioff()
    curr_iter = 0
    to_pil = transforms.ToPILImage()
    total_iters = 0

    py.mkdir(model.output_dir)
    py.mkdir(model.sample_dir)
    py.mkdir(model.checkpoint_dir)

    for epoch in tqdm.trange(opt.epoch_count, opt.epochs, desc="Epoch Loop"):

        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iter = 0
        losses_temp = None
        G_loss_temp = 0
        D_A_loss_temp = 0
        D_B_loss_temp = 0

        if opt.unfreeze_layers_iters > 1:

            iter_num = (epoch - opt.epoch_count) // opt.unfreeze_interval
            print("iter num", iter_num)
            if (epoch - opt.epoch_count) % opt.unfreeze_interval == 0 and iter_num <= opt.unfreeze_layers_iters and iter_num > 0:
                
                model.gradually_unfreeze_layers(opt, sum(G_layers_to_unfreeze[:iter_num+1]), sum(D_layers_to_unfreeze[:iter_num+1]), iter_num)

        for i, batch in tqdm.tqdm(
            enumerate(dataloader),
            desc="Inner Epoch Loop",
            total=len(dataloader.dataset),
        ):

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

        if (
            epoch % opt.save_epoch_freq == 0
        ):  # cache our model every <save_epoch_freq> epochs
            print(
                "saving the model at the end of epoch %d, iters %d"
                % (epoch, total_iters)
            )
            G_losses.append(G_loss_temp / epoch_iter)
            D_A_losses.append(D_A_loss_temp / epoch_iter)
            D_B_losses.append(D_B_loss_temp / epoch_iter)
            model.save_networks(epoch, G_losses, D_A_losses, D_B_losses, opt)
            model.plot_losses(
                epoch, opt.save_epoch_freq, G_losses, D_A_losses, D_B_losses
            )
            model.gen_rec_success(epoch)
