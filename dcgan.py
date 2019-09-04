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
