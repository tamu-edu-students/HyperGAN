import os
from collections import OrderedDict
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import pylib as py
import matplotlib.pyplot as plt
import numpy as np
import io
from PIL import Image
import csv
from . import networks
import itertools
from torchsummary import summary


class BaseModel(ABC):
    """This class is an abstract base class (ABC) for models.
    -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
    -- <set_input>:                     unpack data from dataset and apply preprocessing.
    -- <forward>:                       produce intermediate results.
    -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
    -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the BaseModel class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        When creating your custom class, you need to implement your own initialization.
        In this function, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         define networks used in our training.
            -- self.visual_names (str list):        specify the images that you want to display and save.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        """
        self.opt = opt
        if torch.cuda.is_available():
            opt.cuda = True
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = (
            torch.device("cuda:{}".format(self.gpu_ids[0]))
            if self.gpu_ids
            else torch.device("cpu")
        )  # get device name: CPU or GPU
        self.output_dir = py.join(
            "output", opt.dataroot
        )  # save all the checkpoints to save_dir
        self.checkpoint_dir = py.join(self.output_dir, opt.checkpoints_dir)
        self.sample_dir = py.join(self.output_dir, opt.results_dir)
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        self.lrs = []
        self.image_paths = []
        self.data_length = 0

    def setup(self, opt):
        """The setup sequence is used for restoring a checkpoint from memory to continue training/testing and to bring
        in attributes such as the saved optimizers and learning rates into the current execution

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.G_losses = []  # empty losses for generators
        self.D_A_losses = []  # for discriminator in domain A
        self.D_B_losses = []  ##for discriminator in domain B

        if opt.restore or not self.isTrain:
            print("resume training, or starting testing...")
            epoch_dir = py.join(self.checkpoint_dir, "epoch_{}".format(opt.epoch_count))
            for name in self.model_names:  # Loading in file path of network
                save_filename = "%s_net%s.pth" % (opt.epoch_count, name)
                save_path = py.join(epoch_dir, save_filename)
                net = getattr(self, name)

                # print(net.model[-5:])

                print("loading from: ", save_path)
                net.load_state_dict(
                    torch.load(save_path)
                )  # brings in generator by name in order to apply its restored state to current instance
                if opt.transfer:
                    print("Transfering and Fine Tuning Pretrained Model")
                    for param in net.parameters():
                        param.requires_grad = False

                    if name.find("D") == -1:
                        net.model[-1] = nn.Conv2d(64, opt.output_nc, kernel_size=7)
                    else:
                        net.model[-1] = nn.Conv2d(512, 1, kernel_size=4, padding=1)

                    net.cuda()
                (
                    # summary(net, input_size=(51, 256, 256), device="cuda")
                    # if name.find("G_A2B") != -1
                    # else None
                )

            for optimizer in self.optimizers:  # Loads in optimizer in similar fashion

                if opt.transfer:
                    if optimizer.find("G") != -1:
                        optimizer = torch.optim.Adam(
                            filter(
                                lambda p: p.requires_grad,
                                itertools.chain(
                                    self.G_A2B.parameters(), self.G_B2A.parameters()
                                ),
                            ),
                            lr=opt.lr,
                            betas=(opt.beta_1, 0.999),
                        )

                    elif optimizer.find("D_A") != -1:
                        optimizer = torch.optim.Adam(
                            filter(lambda p: p.requires_grad, self.D_A.parameters()),
                            lr=opt.lr,
                            betas=(0.5, 0.999),
                        )

                    elif optimizer.find("D_B") != -1:
                        optimizer = torch.optim.Adam(
                            filter(lambda p: p.requires_grad, self.D_B.parameters()),
                            lr=opt.lr,
                            betas=(0.5, 0.999),
                        )
                    break

                save_filename = "%s_%s.pth" % (opt.epoch_count, optimizer)
                save_path = py.join(epoch_dir, save_filename)
                optimizer = getattr(self, optimizer)

                print("loading from: ", save_path)
                optimizer.load_state_dict(torch.load(save_path))

            if not opt.transfer:
                for lr in self.lrs:  # Same with learning rate

                    save_filename = "%s_%s.pth" % (opt.epoch_count, lr)
                    save_path = py.join(epoch_dir, save_filename)
                    lr = getattr(self, lr)

                    print("loading from: ", save_path)
                    lr.load_state_dict(torch.load(save_path))

            with open(
                py.join(epoch_dir, "losses_epoch_{}.csv".format(opt.epoch_count)), "r"
            ) as file:  # routing to saved state of previous csv
                csv_reader = csv.reader(file)

                for (
                    row
                ) in (
                    csv_reader
                ):  # restoring losses using csv value stored from previous training session
                    # print(row)
                    # Assuming the order in the CSV file is the same as when writing
                    element1, element2, element3 = row
                    self.G_losses.append(element1)
                    self.D_A_losses.append(element2)
                    self.D_B_losses.append(element3)

            opt.epoch_count += 1  # updating to progress to new epoch
        return self.G_losses, self.D_A_losses, self.D_B_losses  # returning losses

    def save_networks(self, epoch, G_losses, D_A_losses, D_B_losses, opt):
        """Save all the networks to the disk in similar fashion to setup

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
            G/D_A/D_B_losses -- losses from current training session over epochs
            opt -- options
        """
        epoch_dir = py.join(self.checkpoint_dir, "epoch_{}".format(epoch))
        py.mkdir(epoch_dir)
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = "%s_net%s.pth" % (epoch, name)
                save_path = py.join(epoch_dir, save_filename)
                net = getattr(self, name)

                if opt.onnx_export and name == "G_A2B":
                    onnx_save_path = py.join(epoch_dir, "%s_net%s.onnx" % (epoch, name))

                    dummy_input = torch.randn(
                        1, opt.input_nc, opt.crop_size, opt.crop_size
                    ).to("cuda")
                    torch.onnx.export(net, dummy_input, onnx_save_path, verbose=True)

                print("saving to, ", save_path)
                torch.save(net.state_dict(), save_path)

        for optimizer in self.optimizers:

            save_filename = "%s_%s.pth" % (epoch, optimizer)
            save_path = py.join(epoch_dir, save_filename)
            optimizer = getattr(self, optimizer)

            torch.save(optimizer.state_dict(), save_path)

        for lr in self.lrs:
            save_filename = "%s_%s.pth" % (epoch, lr)
            save_path = py.join(epoch_dir, save_filename)
            lr = getattr(self, lr)

            print("saving to, ", save_path)
            torch.save(lr.state_dict(), save_path)

        # if opt.onnx_export:
        #     save_filename = '%s_net%s.onnx' % (epoch, name)
        #     save_path = py.join(epoch_dir, save_filename)
        #     net = getattr(self, name)

        rows = zip(G_losses, D_A_losses, D_B_losses)

        with open(
            py.join(epoch_dir, "losses_epoch_{}.csv".format(epoch)), "w", newline=""
        ) as file:
            csv_writer = csv.writer(file)
            csv_writer.writerows(rows)

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new model-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def create_networks(self, input):
        pass

    @abstractmethod
    def update_learning_rate():
        pass

    @abstractmethod
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        pass

    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    @abstractmethod
    def get_visuals(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(
                    getattr(self, "loss_" + name)
                )  # float(...) works for both scalar tensor and float number
        return errors_ret

    def plot_losses(self, epoch, save_freq, errors_G, errors_D_A, errors_D_B):
        """
        Plots the losses in the same directory as the epoch checkpoint in order to provide a visualizer
        """
        plt.clf()
        tot_epochs = np.arange(
            1, len(errors_G) + 1, save_freq
        )  # creating x axis for plot
        plt.plot(
            tot_epochs, errors_G, label="Generator Loss"
        )  # plotting loss components over the amount of epochs trainined for each network
        plt.plot(tot_epochs, errors_D_A, label="Discriminator Loss: Shadow")
        plt.plot(tot_epochs, errors_D_B, label="Discriminator Loss: Nonshadow")
        plt.xlabel("Epoch")  # labeling the plot
        plt.ylabel("Loss Value")
        plt.title("Loss for Subnet Over Epochs")
        plt.legend()

        buffer = io.BytesIO()

        # Save the Matplotlib plot to the BytesIO buffer as a PIL image
        plt.savefig(buffer, format="png")
        buffer.seek(0)  # Move the buffer cursor to the beginning

        Image.open(buffer).save(
            py.join(self.checkpoint_dir, "losses_{}.png".format(epoch))
        )
        # Open the PIL image from the BytesIO buffer

    def gradually_unfreeze_layers(
        self, opt, G_layer_count, D_layer_count, unfreeze_iter
    ):
        for name in self.model_names:
            net = getattr(self, name)
            print()
            print("Unfreezing {} layers for {}".format( (G_layer_count if name.find('G') != -1 else D_layer_count), name))
            if unfreeze_iter == 0:
                for param in net.parameters():
                    param.requires_grad = False

            if name.find("G") != -1:
                for param in net.model[-G_layer_count:].parameters():
                    param.requires_grad = True
                print("Trainable layers for {}:".format(name))
                print(net.model[-G_layer_count:], "\n")
                total_G_params, trainable_G_params = self.count_parameters(net)
                print("For {}, total parameters are: {} and trainable parameters are: {}".format(name, total_G_params, trainable_G_params), "\n")


            if name.find("D") != -1:
                for param in net.model[-D_layer_count:].parameters():
                    param.requires_grad = True
                print("Trainable layers for {}:".format(name))
                print(net.model[-D_layer_count:], "\n")
                total_D_params, trainable_D_params = self.count_parameters(net)
                print("For {}, total parameters are: {} and trainable parameters are: {}".format(name, total_D_params, trainable_D_params), "\n")

        self.optimizer_G = torch.optim.Adam(
            filter(
                lambda p: p.requires_grad,
                itertools.chain(self.G_A2B.parameters(), self.G_B2A.parameters()),
            ),
            lr=opt.lr,
            betas=(opt.beta_1, 0.999),
        )
        self.optimizer_D_A = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.D_A.parameters()),
            lr=opt.lr,
            betas=(0.5, 0.999),
        )
        self.optimizer_D_B = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.D_B.parameters()),
            lr=opt.lr,
            betas=(0.5, 0.999),
        )

    def count_parameters(self, model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total_params, trainable_params
