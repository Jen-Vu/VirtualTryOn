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

# Create an object of Generater
netG = Generator()
netG.apply(weights_init)


# Define the Descriminator

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
                nn.Conv2d(3, 64, 4, 2, 1, bias = False),
                nn.LeakyReLU(0.2, inplace = True),
                nn.Conv2d(64, 128, 4, 2, 1, bias = False),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace = True),
                nn.Conv2d(128, 256, 4, 2, 1, bias = False),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace = True),
                nn.Conv2d(256, 512, 4, 2, 1, bias = False),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace = True),
                nn.Conv2d(512, 1, 4, 1, 0, bias = False),
                # use Sigmoid to break the linearity and normalise the result to 0 to 1
                nn.Sigmoid()
        )

    # For forward propagation
    def forward(self, input):
        output = self.main(input)
        return output.view(-1)

# Create an object of Generater
netD = Discriminator()
netD.apply(weights_init)

# Binary Cross Entropy loss for DCGANs
criterion = nn.BCELoss()
# Create optimizers
optimizerD = optim.Adam(netD.parameters(),
                        lr = 0.0002,
                        betas = (0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(),
                        lr = 0.0002,
                        betas = (0.5, 0.999))

for epoch in range(25):
    for i, data in enumerate(dataloader, 0):
        # 1st Step: Updating the weights of the neural network of the Discriminator
        netD.zero_grad()

        # Training the discriminator with a real image of the dataset
        real, _ = data
        input = Variable(real)
        # to create a Variable of Ones of size input.size()[0]
        target = Variable(torch.ones(input.size()[0]))
        output = netD(input)
        errD_real = criterion(output, target)

        # Training the discriminator with a fake image generated by the generator
        noise = Variable(torch.randn(input.size()[0], 100, 1, 1)) # make a random input vector (noise) of the generator.
        fake = netG(noise) # forward propagate random input vector into the neural network of the generator to get some fake generated images.
        target = Variable(torch.zeros(input.size()[0]))
        output = netD(fake.detach())
        errD_fake = criterion(output, target)

        # Backpropagating the total error
        errD = errD_real + errD_fake # Compute the total error of the discriminator.
        errD.backward() # Backpropagate the loss error by computing the gradients of the total error with respect to the weights of the discriminator.
        optimizerD.step() # Apply the optimizer to update the weights according to how much they are responsible for the loss error of the discriminator.

        # 2nd Step: Updating the weights of the neural network of the Generator

        # Initialize to 0 the gradients of the generator with respect to the weights.
        netG.zero_grad()
        target = Variable(torch.ones(input.size()[0]))
        # Forward propagate the fake generated images into the neural network of the discriminator to get the prediction (a value between 0 and 1).
        output = netD(fake)
        # Compute the loss between the prediction (output between 0 and 1) and the target (equal to 1).
        errG = criterion(output, target)
        # Backpropagate the loss error by computing the gradients of the total error with respect to the weights of the generator.
        errG.backward()
        # Apply the optimizer to update the weights according to how much they are responsible for the loss error of the generator.
        optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f' % (epoch, 25, i, len(dataloader), errD.data, errG.data))
        if i % 100 == 0:
            vutils.save_image(real, '%s/real_samples_epoch_%03d.png' % ("./results", epoch), normalize = True)
            fake = netG(noise)
            vutils.save_image(fake.data, '%s/fake_samples_epoch_%03d.png' % ("./results", epoch), normalize = True)
