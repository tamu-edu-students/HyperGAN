import os
from collections import OrderedDict
from abc import ABC, abstractmethod
import torch
import pylib as py
import matplotlib.pyplot as plt
import numpy as np
import io
from PIL import Image
import csv

class BaseModel(ABC):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
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
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
        self.output_dir = py.join('output', opt.dataroot)  # save all the checkpoints to save_dir
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
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.G_losses = []
        self.D_A_losses = []
        self.D_B_losses = []

        if opt.restore or not self.isTrain:
            print('resume training, or starting testing...')
            epoch_dir = py.join(self.checkpoint_dir, 'epoch_{}'.format(opt.epoch_count))
            for name in self.model_names:
                save_filename = '%s_net%s.pth' % (opt.epoch_count, name)
                save_path = py.join(epoch_dir, save_filename)
                net = getattr(self, name)
  
                print("loading from: ", save_path)
                net.load_state_dict(torch.load(save_path))
            
            for optimizer in self.optimizers:
                
                save_filename = '%s_%s.pth' % (opt.epoch_count, optimizer)
                save_path = py.join(epoch_dir, save_filename)
                optimizer = getattr(self, optimizer)
  
                print("loading from: ", save_path)
                optimizer.load_state_dict(torch.load(save_path))

            for lr in self.lrs:
                save_filename = '%s_%s.pth' % (opt.epoch_count, lr)
                save_path = py.join(epoch_dir, save_filename)
                lr = getattr(self, lr)

                print("loading from: ", save_path)
                lr.load_state_dict(torch.load(save_path))

            with open(py.join(epoch_dir, "losses_epoch_{}.csv".format(opt.epoch_count)), 'r') as file:
                csv_reader = csv.reader(file)
                        
                for row in csv_reader:
                    print(row)
                    # Assuming the order in the CSV file is the same as when writing
                    element1, element2, element3 = row
                    self.G_losses.append(element1)
                    self.D_A_losses.append(element2)
                    self.D_B_losses.append(element3)
            
            opt.epoch_count += 1 # updating to progress to new epoch
            print("returningggg")
        return self.G_losses, self.D_A_losses, self.D_B_losses  
        
    
    def save_networks(self, epoch, G_losses, D_A_losses, D_B_losses, opt):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        epoch_dir = py.join(self.checkpoint_dir, 'epoch_{}'.format(epoch))
        py.mkdir(epoch_dir)
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net%s.pth' % (epoch, name)
                save_path = py.join(epoch_dir, save_filename)
                net = getattr(self, name)

                print("saving to, ", save_path)
                torch.save(net.state_dict(), save_path)

        for optimizer in self.optimizers:
            
            save_filename = '%s_%s.pth' % (epoch, optimizer)
            save_path = py.join(epoch_dir, save_filename)
            optimizer = getattr(self, optimizer)

            torch.save(optimizer.state_dict(), save_path)

        for lr in self.lrs:
            save_filename = '%s_%s.pth' % (epoch, lr)
            save_path = py.join(epoch_dir, save_filename)
            lr = getattr(self, lr)

            print("saving to, ", save_path)
            torch.save(lr.state_dict(), save_path)

        rows = zip(G_losses, D_A_losses, D_B_losses)

        with open(py.join(epoch_dir, "losses_epoch_{}.csv".format(epoch)), 'w', newline='') as file:
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
                errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret
    
    def plot_losses(self, epoch, save_freq, errors_G, errors_D_A, errors_D_B):

        plt.clf()
        band_numbers = np.arange(1, len(errors_G)+1, save_freq)
        plt.plot(band_numbers, errors_G,  label='Generator Loss')
        plt.plot(band_numbers, errors_D_A, label='Discriminator Loss: Shadow')
        plt.plot(band_numbers, errors_D_B, label='Discriminator Loss: Nonshadow')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Value')
        plt.title('Loss for Subnet Over Epochs')
        plt.legend()

        buffer = io.BytesIO()

        # Save the Matplotlib plot to the BytesIO buffer as a PIL image
        plt.savefig(buffer, format='png')
        buffer.seek(0)  # Move the buffer cursor to the beginning
        
        Image.open(buffer).save(py.join(self.checkpoint_dir, 'losses_{}.png'.format(epoch)))
        # Open the PIL image from the BytesIO buffer
