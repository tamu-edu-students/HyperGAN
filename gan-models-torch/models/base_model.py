import os
from collections import OrderedDict
from abc import ABC, abstractmethod
import torch

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
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # save all the checkpoints to save_dir
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        self.lrs = []
        self.image_paths = []
        self.metric = 0  # used for learning rate policy 'plateau'

    def setup(self, opt):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if opt.restore:
            print
            'resume training:'
            self.G_A2B.load_state_dict(torch.load('output/netG_A2B.pth'))
            self.G_B2A.load_state_dict(torch.load('output/netG_B2A.pth'))
            self.D_A.load_state_dict(torch.load('output/netD_A.pth'))
            self.D_B.load_state_dict(torch.load('output/netD_B.pth'))

            self.optimizer_G.load_state_dict(torch.load('output/optimizer_G.pth'))
            self.optimizer_D_A.load_state_dict(torch.load('output/optimizer_D_A.pth'))
            self.optimizer_D_B.load_state_dict(torch.load('output/optimizer_D_B.pth'))

            self.lr_scheduler_G.load_state_dict(torch.load('output/lr_scheduler_G.pth'))
            self.lr_scheduler_D_A.load_state_dict(torch.load('output/lr_scheduler_D_A.pth'))
            self.lr_scheduler_D_B.load_state_dict(torch.load('output/lr_scheduler_D_B.pth'))
    
    def save_networks(self, opt):
        torch.save(self.A2B.state_dict(), 'output/netG_A2B.pth')
        torch.save(self.G_B2A.state_dict(), 'output/netG_B2A.pth')
        torch.save(self.D_A.state_dict(), 'output/netD_A.pth')
        torch.save(self.D_B.state_dict(), 'output/netD_B.pth')

        torch.save(self.optimizer_G.state_dict(), 'output/optimizer_G.pth')
        torch.save(self.optimizer_D_A.state_dict(), 'output/optimizer_D_A.pth')
        torch.save(self.optimizer_D_B.state_dict(), 'output/optimizer_D_B.pth')

        torch.save(self.lr_scheduler_G.state_dict(), 'output/lr_scheduler_G.pth')
        torch.save(self.lr_scheduler_D_A.state_dict(), 'output/lr_scheduler_D_A.pth')
        torch.save(self.lr_scheduler_D_B.state_dict(), 'output/lr_scheduler_D_B.pth')
    
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

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret