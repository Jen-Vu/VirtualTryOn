#Deep Convolutional GANs for Image Generation Using Textual Information

# Han Jie

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

# Set hyperparameters
batchSize = 64
imageSize = 64

# Create transformations
# Create a list of transformations (scaling, tensor conversion, normalization) to apply to the input images.
transformation = transforms.Compose([transforms.Scale(imageSize), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]) 


# Load dataset
dataset = dset.CIFAR10(root = './data', download = True, transform = transformation)
# use dataLoader to get the images of the training set batch by batch.
dataloader = torch.utils.data.DataLoader(dataset, batch_size = batchSize, shuffle = True, num_workers = 2)

# Define weight initialisation fcuntion, input being uninitialisaed neural network, nnet
def weights_init(nnet):
    classname = nnet.__class__.__name__
    if classname.find('Conv') != -1:
        nnet.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nnet.weight.data.normal_(1.0, 0.02)
        nnet.bias.data.fill_(0)

# Define Generator Class
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # Inverse convolution
        self.main = nn.Sequential(
                # size of input: 100
                # size of out channels: 512
                # size of the kernel: 4 x 4
                # stride: 1
                # padding: 1
                # bias: false
                # First Inverse Conv
                nn.ConvTranspose2d(100, 512, 4, 1, 0, bias = False),
                nn.BatchNorm2d(512),
                nn.ReLU(True),
                # 512 is the previous output and this new input
                # Second Inverse Conv
                nn.ConvTranspose2d(512, 256, 4, 2, 1, bias = False),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                # Third Inverse Conv
                nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                # Fourth Inverse Conv
                nn.ConvTranspose2d(128, 64, 4, 2, 1, bias = False),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                # out_channel = 3 meaning 3 colour channels
                nn.ConvTranspose2d(64, 3, 4, 2, 1, bias = False),
                # render it to be between -1 to 1 using inverse tan
                nn.Tanh()
        )

    # For forward propagation
    # Input: random vector noise to generate image
    def forward(self, input):
        # Feed the NN with input and get the output, i.e. 3 channels of the image
        output = self.main(input)
        return output

# Create an instance of Generater
netG = Generator()
netG.apply(weights_init)