import torch.nn as nn
import torch.nn.functional as F
import torch

class ResidualBlock(nn.Module):
    # Constructor method with in_features as a parameter
    def __init__(self, in_features):
        # Call the constructor of the parent class (nn.Module)
        super(ResidualBlock, self).__init__()

        # Define a convolutional block as a list
        conv_block = [
            # Pad the input tensor with reflection padding of size 1
            nn.ReflectionPad2d(1),
            # 3x3 convolution with in_features input channels and in_features output channels
            nn.Conv2d(in_features, in_features, 3),
            # Instance normalization on the convolutional output
            nn.InstanceNorm2d(in_features),
            # Apply the ReLU activation function in place
            nn.ReLU(inplace=True),
            # Pad the output of the previous layer with reflection padding of size 1
            nn.ReflectionPad2d(1),
            # 3x3 convolution with in_features input channels and in_features output channels
            nn.Conv2d(in_features, in_features, 3),
            # Instance normalization on the convolutional output
            nn.InstanceNorm2d(in_features)
        ]

        # Create a sequential block using the convolutional block list
        self.conv_block = nn.Sequential(*conv_block)

    # Forward method defining the forward pass of the block
    def forward(self, x):
        # Return the sum of the input tensor and the output of the convolutional block
        return x + self.conv_block(x)

class Generator(nn.Module):
    # Constructor method with input_nc, output_nc, and n_residual_blocks as parameters
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        # Call the constructor of the parent class (nn.Module)
        super(Generator, self).__init__()

        # Initial convolution block
        model = [
            # Pad the input tensor with reflection padding of size 3
            nn.ReflectionPad2d(3),
            # 7x7 convolution with input_nc input channels and 64 output channels
            nn.Conv2d(input_nc, 64, 7),
            # Instance normalization on the convolutional output
            nn.InstanceNorm2d(64),
            # Apply the ReLU activation function in place
            nn.ReLU(inplace=True)
        ]

        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            # 3x3 convolution with in_features input channels and out_features output channels, stride=2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                # Instance normalization on the convolutional output
                nn.InstanceNorm2d(out_features),
                # Apply the ReLU activation function in place
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks
        for _ in range(n_residual_blocks):
            # Add ResidualBlock with in_features as a parameter to the model
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            # 3x3 transpose convolution with in_features input channels and out_features output channels, stride=2
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                # Instance normalization on the convolutional output
                nn.InstanceNorm2d(out_features),
                # Apply the ReLU activation function in place
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model += [
            # Pad the input tensor with reflection padding of size 3
            nn.ReflectionPad2d(3),
            # 7x7 convolution with 64 input channels and output_nc output channels
            nn.Conv2d(64, output_nc, 7)
            # Tanh activation function is typically used to scale outputs between -1 and 1
            # nn.Tanh()
        ]

        # Create a sequential model using the constructed list
        self.model = nn.Sequential(*model)

    # Forward method defining the forward pass of the generator
    def forward(self, x):
        # Return the sum of the generator model output and the input tensor, scaled by Tanh activation
        return (self.model(x) + x).tanh()

class Generator_F2S(nn.Module):
    # Constructor method with input_nc, output_nc, and n_residual_blocks as parameters
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        # Call the constructor of the parent class (nn.Module)
        super(Generator_F2S, self).__init__()

        # Initial convolution block
        model = [
            # Pad the input tensor with reflection padding of size 3
            nn.ReflectionPad2d(3),
            # 7x7 convolution with input_nc+1 input channels and 64 output channels
            nn.Conv2d(input_nc+1, 64, 7),
            # Instance normalization on the convolutional output
            nn.InstanceNorm2d(64),
            # Apply the ReLU activation function in place
            nn.ReLU(inplace=True)
        ]

        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            # 3x3 convolution with in_features input channels and out_features output channels, stride=2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                # Instance normalization on the convolutional output
                nn.InstanceNorm2d(out_features),
                # Apply the ReLU activation function in place
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks
        for _ in range(n_residual_blocks):
            # Add ResidualBlock with in_features as a parameter to the model
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            # 3x3 transpose convolution with in_features input channels and out_features output channels, stride=2
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                # Instance normalization on the convolutional output
                nn.InstanceNorm2d(out_features),
                # Apply the ReLU activation function in place
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model += [
            # Pad the input tensor with reflection padding of size 3
            nn.ReflectionPad2d(3),
            # 7x7 convolution with 64 input channels and output_nc output channels
            nn.Conv2d(64, output_nc, 7)
            # Tanh activation function is typically used to scale outputs between -1 and 1
            # nn.Tanh()
        ]

        # Create a sequential model using the constructed list
        self.model = nn.Sequential(*model)

    # Forward method defining the forward pass of the generator
    def forward(self, x, mask):
        # Return the sum of the generator model output and the input tensor, scaled by Tanh activation
         return (self.model(torch.cat((x, mask), 1)) + x).tanh() #(min=-1, max=1) #just learn a residual

# Define a Discriminator class that inherits from nn.Module
class Discriminator(nn.Module):
    # Constructor method with input_nc as a parameter
    def __init__(self, input_nc):
        # Call the constructor of the parent class (nn.Module)
        super(Discriminator, self).__init__()

        # A series of convolutional layers with leaky ReLU activations

        # Convolutional layer with input_nc input channels, 64 output channels, 4x4 kernel, stride=2, padding=1
        model = [
            nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
            # Leaky ReLU activation with a negative slope of 0.2, inplace operation
            nn.LeakyReLU(0.2, inplace=True)
        ]

        # Convolutional layer with 64 input channels, 128 output channels, 4x4 kernel, stride=2, padding=1
        model += [
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            # Instance normalization on the convolutional output
            nn.InstanceNorm2d(128),
            # Leaky ReLU activation with a negative slope of 0.2, inplace operation
            nn.LeakyReLU(0.2, inplace=True)
        ]

        # Convolutional layer with 128 input channels, 256 output channels, 4x4 kernel, stride=2, padding=1
        model += [
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            # Instance normalization on the convolutional output
            nn.InstanceNorm2d(256),
            # Leaky ReLU activation with a negative slope of 0.2, inplace operation
            nn.LeakyReLU(0.2, inplace=True)
        ]

        # Convolutional layer with 256 input channels, 512 output channels, 4x4 kernel, padding=1
        model += [
            nn.Conv2d(256, 512, 4, padding=1),
            # Instance normalization on the convolutional output
            nn.InstanceNorm2d(512),
            # Leaky ReLU activation with a negative slope of 0.2, inplace operation
            nn.LeakyReLU(0.2, inplace=True)
        ]

        # Fully connected convolutional (FCN) layer for classification
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        # Create a sequential model using the constructed list
        self.model = nn.Sequential(*model)

    # Forward method defining the forward pass of the discriminator
    def forward(self, x):
        x = self.model(x)
        # Average pooling and flatten the output for global average pooling
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)  # global avg pool
