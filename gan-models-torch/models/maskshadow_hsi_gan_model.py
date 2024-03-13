import itertools
from .base_model import BaseModel
from . import networks
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset.maskshadow_dataset import MaskImageDataset
import pylib as py
import random
from PIL import Image, ImageDraw
from util.util import (
    LambdaLR,
    weights_init_normal,
    ReplayBuffer,
    QueueMask,
    mask_generator,
    mod_to_pil,
    tensor2spectral,
    mod_to_spectral,
)
from util.constants import highlight_points
from hyperspectral.util import utils


class MaskShadowHsiGANModel(BaseModel):
    def __init__(self, opt):
        BaseModel.__init__(self, opt)  # calling constructor of super class
        self.loss_names = [
            "D_A",
            "G_A2B",
            "cycle_ABA",
            "idt_A",
            "D_B",
            "G_B2A",
            "cycle_BAB",
            "idt_B",
        ]  # setting up loss names
        self.create_networks(opt)  # creating networks upon startup

    def setup(self, opt):
        super().setup(opt)

        self.mask_queue = QueueMask(
            self.data_length / 4
        )  # Queue for storing masks with finite storage to prioritize mask generation improvement
        print(self.data_length / 4)  # printing the length

    def create_networks(self, opt):

        if self.isTrain:
            self.model_names = [
                "G_A2B",
                "G_B2A",
                "D_A",
                "D_B",
            ]  # initializing generator and discriminator list for getattr()
            self.G_A2B = networks.Generator(
                opt.input_nc, opt.output_nc
            )  # initializing subnets for image to image translation
            self.G_B2A = networks.Generator_F2S(opt.output_nc, opt.input_nc)
            self.D_A = networks.Discriminator(opt.input_nc)
            self.D_B = networks.Discriminator(opt.output_nc)

            if (
                opt.cuda
            ):  # detecting GPU on runtime for models to be evaluated using GPU compute
                print("Using GPU")
                self.G_A2B.cuda()
                self.G_B2A.cuda()
                self.D_A.cuda()
                self.D_B.cuda()

            self.G_A2B.apply(weights_init_normal)  # initializing baseline weights
            self.G_B2A.apply(weights_init_normal)
            self.D_A.apply(weights_init_normal)
            self.D_B.apply(weights_init_normal)

            self.criterionGAN = torch.nn.MSELoss()  # define GAN loss.
            self.criterionCycle = (
                torch.nn.L1Loss()
            )  # cyclic loss after passing through both generators
            self.criterionIdentity = (
                torch.nn.L1Loss()
            )  # identity loss for using wrong generator on image from another domain
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(
                itertools.chain(self.G_A2B.parameters(), self.G_B2A.parameters()),
                lr=opt.lr,
                betas=(opt.beta_1, 0.999),
            )
            self.optimizer_D_A = torch.optim.Adam(
                self.D_A.parameters(), lr=opt.lr, betas=(0.5, 0.999)
            )
            self.optimizer_D_B = torch.optim.Adam(
                self.D_B.parameters(), lr=opt.lr, betas=(0.5, 0.999)
            )
            self.optimizers.append("optimizer_G")
            self.optimizers.append(
                "optimizer_D_A"
            )  # appending optimizers by name for saving and loading
            self.optimizers.append(
                "optimizer_D_B"
            )  # getattr() will be called to reference the object
            self.lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer_G,
                lr_lambda=LambdaLR(opt.epochs, opt.epoch_count, opt.epoch_decay).step,
            )
            self.lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer_D_A,
                lr_lambda=LambdaLR(opt.epochs, opt.epoch_count, opt.epoch_decay).step,
            )
            self.lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer_D_B,
                lr_lambda=LambdaLR(opt.epochs, opt.epoch_count, opt.epoch_decay).step,
            )
            self.lrs.append(
                "lr_scheduler_G"
            )  # appending learning rate schedulars by name for saving and loading
            self.lrs.append(
                "lr_scheduler_D_A"
            )  # getattr() will be called to reference the object
            self.lrs.append("lr_scheduler_D_B")

        else:  # during test time, only load Gs
            self.model_names = [
                "G_A2B",
                "G_B2A",
            ]  # for testing, only generators are required
            self.G_A2B = networks.Generator(opt.input_nc, opt.output_nc)
            self.G_B2A = networks.Generator_F2S(opt.output_nc, opt.input_nc)

            if opt.cuda:
                print("Using GPU")
                self.G_A2B.cuda()
                self.G_B2A.cuda()  # transferring computation to GPU

            self.G_A2B.apply(weights_init_normal)  # initializing normal weights
            self.G_B2A.apply(weights_init_normal)

        Tensor = (
            torch.cuda.FloatTensor if opt.cuda else torch.Tensor
        )  # Using float tensor given GPU resource
        self.input_A = Tensor(
            opt.batch_size, opt.input_nc, opt.crop_size, opt.crop_size
        )  # input tensor for data unpacking - Domain A
        self.input_B = Tensor(
            opt.batch_size, opt.output_nc, opt.crop_size, opt.crop_size
        )  # input tensor for data unpacking - Domain B
        self.target_real = torch.autograd.Variable(
            Tensor(opt.batch_size).fill_(1.0), requires_grad=False
        )  # tensor for a real decision - all 1s
        self.target_fake = torch.autograd.Variable(
            Tensor(opt.batch_size).fill_(0.0), requires_grad=False
        )  # tensor for a fake decision - all 0s
        self.mask_non_shadow = torch.autograd.Variable(
            Tensor(opt.batch_size, 1, opt.crop_size, opt.crop_size).fill_(-1.0),
            requires_grad=False,
        )  # -1.0 non-shadow

        self.fake_A_buffer = ReplayBuffer()
        self.fake_B_buffer = ReplayBuffer()

        print("networks created: ", self.model_names)

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument(
                "--lambda_A",
                type=float,
                default=10.0,
                help="weight for cycle loss (A -> B -> A)",
            )
            parser.add_argument(
                "--lambda_B",
                type=float,
                default=10.0,
                help="weight for cycle loss (B -> A -> B)",
            )
            parser.add_argument(
                "--lambda_identity",
                type=float,
                default=0.5,
                help="use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1",
            )

        return parser

    def update_learning_rate(self):
        for lr in self.lrs:
            getattr(self, lr).step()

    def mask_init(self, length):
        self.mask_queue = QueueMask(length / 4)

    def set_input(self, batch):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        self.real_A = torch.autograd.Variable(self.input_A.copy_(batch["A"]))
        self.real_B = torch.autograd.Variable(self.input_B.copy_(batch["B"]))
        return None

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.G_A2B(self.real_A)  # G_A(A)
        self.computed_mask = mask_generator(
            self.real_A, self.fake_B, isHyper=True
        )  # processing tensors for mask generation via thresholding
        self.mask_queue.insert(
            self.computed_mask
        )  # inserting generated mask into queue

        self.rec_A = self.G_B2A(self.fake_B, self.mask_queue.last_item())  # G_B(G_A(A))
        self.rand_guide_mask = (
            self.mask_queue.rand_item()
        )  # using random mask from queue for guidance
        self.fake_A = self.G_B2A(self.real_B, self.rand_guide_mask)  # G_B(B)

        self.rec_B = self.G_A2B(self.fake_A)  # G_A(G_B(B))

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, self.target_real)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, self.target_fake)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_A = self.fake_A_buffer.push_and_pop(self.fake_A)
        self.loss_D_A = self.backward_D_basic(self.D_A, self.real_A, fake_A)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_B = self.fake_B_buffer.push_and_pop(self.fake_B)
        self.loss_D_B = self.backward_D_basic(self.D_B, self.real_B, fake_B)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.G_A2B(self.real_B)
            self.loss_idt_A = (
                self.criterionIdentity(self.idt_A, self.real_B) * lambda_B * lambda_idt
            )
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.G_B2A(self.real_A, self.mask_non_shadow)
            self.loss_idt_B = (
                self.criterionIdentity(self.idt_B, self.real_A) * lambda_A * lambda_idt
            )
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A2B = self.criterionGAN(self.D_B(self.fake_B), self.target_real)

        # GAN loss D_B(G_B(B))
        self.loss_G_B2A = self.criterionGAN(self.D_A(self.fake_A), self.target_real)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_ABA = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_BAB = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss and calculate gradients
        self.loss_G = (
            self.loss_G_A2B
            + self.loss_G_B2A
            + self.loss_cycle_ABA
            + self.loss_cycle_BAB
            + self.loss_idt_A
            + self.loss_idt_B
        )

        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()  # compute fake images and reconstruction images.
        # G_A and G_B
        # self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()  # calculate gradients for G_A and G_B
        self.optimizer_G.step()  # update G_A and G_B's weights
        # D_A and D_B
        # self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D_A.zero_grad()  # set D_A and D_B's gradients to zero
        self.optimizer_D_B.zero_grad()  # set D_A and D_B's gradients to zero
        self.backward_D_A()  # calculate gradients for D_A
        self.backward_D_B()  # calculate graidents for D_B
        self.optimizer_D_A.step()  # update D_A and D_B's weights
        self.optimizer_D_B.step()  # update D_A and D_B's weights

    def get_visuals(self, epoch_iter, epoch=0):

        images = []
        num_rows = 0
        num_columns = 0

        if self.isTrain:
            images.append(mod_to_pil(self.real_A, True))
            images.append(mod_to_pil(self.fake_B, True))
            images.append(mod_to_pil(self.rec_A, True))
            images.append(mod_to_pil(self.computed_mask, False))
            images.append(mod_to_pil(self.real_B, True))
            images.append(mod_to_pil(self.fake_A, True))
            images.append(mod_to_pil(self.rec_B, True))
            images.append(mod_to_pil(self.rand_guide_mask, False))
            num_rows = 2
            num_columns = 4
        else:
            images.append(mod_to_pil(self.real_A, True))
            images.append(mod_to_pil(self.fake_B, True))
            images.append(mod_to_pil(self.computed_mask, False))
            num_rows = 1
            num_columns = 3

        image_width, image_height = images[0].size
        output_width = num_columns * image_width
        output_height = num_rows * image_height

        output_image = Image.new("RGB", (output_width, output_height))
        # Paste the individual PIL images onto the output image
        for i, image in enumerate(images):
            row = i // num_columns
            col = i % num_columns
            x_offset = col * image_width
            y_offset = row * image_height
            output_image.paste(image, (x_offset, y_offset))

        # Save the combined image as a PNG file
        if self.isTrain:
            output_image.save(
                py.join(
                    self.sample_dir, "epoch-{}-iter-{}.jpg".format(epoch, epoch_iter)
                )
            )
        else:
            output_image.save(py.join(self.sample_dir, "img-{}.jpg".format(epoch_iter)))
            return (
                mod_to_pil(self.real_A, True),
                mod_to_pil(self.fake_B, True),
                mod_to_spectral(self.real_A),
                mod_to_spectral(self.fake_B),
            )

    def save_mask(self):

        binMask = mod_to_pil(self.rand_guide_mask, False)
        binMask = binMask.convert('L')
        img = binMask.point(lambda p: p > 128 and 255)
        img.save(py.join(self.sample_dir, "img-mask.jpg"))
        return img

    def expand_dataset(self):

        hyper_shadowed = []

        for i in range(int(self.data_length / 4)):  # iterating through queue
            self.rand_guide_mask = self.mask_queue.rand_item()  # selecting random mask
            self.fake_A = self.G_B2A(
                self.real_B, self.rand_guide_mask
            )  # G_B(B) using sampled mask
            print(i)
            images = []  # apending results
            images.append(mod_to_pil(self.real_B))
            images.append(mod_to_pil(self.fake_A))
            images.append(mod_to_pil(self.rand_guide_mask))
            num_rows = 1
            num_columns = 3  # arranging GT, fake shadowed A, and guidance mask

            image_width, image_height = images[0].size
            output_width = num_columns * image_width
            output_height = num_rows * image_height

            output_image = Image.new("RGB", (output_width, output_height))

            # Paste the individual PIL images onto the output image
            for i, image in enumerate(images):
                row = i // num_columns
                col = i % num_columns
                x_offset = col * image_width
                y_offset = row * image_height
                output_image.paste(image, (x_offset, y_offset))
            hyper_shadowed.append(output_image)

        iter = 0
        for hs in hyper_shadowed:  # iterating through all sampled artificial images
            iter += 1
            hs.save(py.join(self.sample_dir, "img-{}.jpg".format(iter)))

    def gen_rec_success(self, epoch):

        images = []
        ground_truth = self.real_A
        shadowed = self.real_A
        reconstructed = self.fake_B

        # print("gt shape:", ground_truth.shape)
        # print("shadowed shape: ", shadowed.shape)
        # print("reconstructed shape:", reconstructed.shape)

        ground_truth_rgb = mod_to_pil(ground_truth, True)
        shadowed_rgb = mod_to_pil(shadowed, True)
        reconstructed_rgb = mod_to_pil(reconstructed, True)

        draw_gt = ImageDraw.Draw(ground_truth_rgb)
        draw_shadowed = ImageDraw.Draw(shadowed_rgb)
        draw_reconstructed = ImageDraw.Draw(reconstructed_rgb)

        # Get the width and height of the image
        width, height = shadowed_rgb.size

        # utils.highlight_selector(shadowed_rgb, reconstructed_rgb)

        # Generate random points and overlay boxes
        for i in range(len(highlight_points)):
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)

            images.append(
                utils.spectral_plot(
                    tensor2spectral(ground_truth, x, y),
                    tensor2spectral(shadowed, x, y),
                    tensor2spectral(reconstructed, x, y),
                    highlight_points[i],
                )
            )

            box_size = 10  # Adjust the size of the box as needed
            draw_gt.rectangle(
                [x, y, x + box_size, y + box_size], outline=highlight_points[i], width=3
            )
            draw_shadowed.rectangle(
                [x, y, x + box_size, y + box_size], outline=highlight_points[i], width=3
            )
            draw_reconstructed.rectangle(
                [x, y, x + box_size, y + box_size], outline=highlight_points[i], width=3
            )  # You can change the box color and width

        images.append(ground_truth_rgb)
        images.append(shadowed_rgb)
        images.append(reconstructed_rgb)

        num_rows = 3
        num_columns = 3

        image_width, image_height = images[0].size
        output_width = num_columns * image_width
        output_height = num_rows * image_height

        output_image = Image.new("RGB", (output_width, output_height))
        # Paste the individual PIL images onto the output image
        for i, image in enumerate(images):
            row = i // num_columns
            col = i % num_columns
            x_offset = col * image_width
            y_offset = row * image_height
            output_image.paste(image, (x_offset, y_offset))

        # Save the combined image as a PNG file

        output_image.save(
            py.join(self.sample_dir, "img-reconstruction-{}.jpg".format(epoch))
        )
        # Save the modified image
        shadowed_rgb.save("output_image.jpg")
