### Python file that contains the different types of discriminators used PatchGAN, Critic, etc.

# Import the necessary libraries

# Import pytorch for creating neural networks
import torch
# Import nn module for creating neural networks
import torch.nn as nn
# Import the blocks used to create the discriminator
from .Blocks import *



# Define the PatchGAN discriminator
class CycleGAN_Discriminator(nn.Module):
    '''
    Discriminator as introduced in CycleGAN paper.
    Discriminator Class. Takes an image as input and outputs a matrix of values classifying if certain patches of image are real or fake.
    Inputs: input_channels -> The number of channels in the input (e.g 3 for RBG image)
            initial_channels -> The number of channels for the first layer of the discriminator
    '''
    
    # Define the constructor
    def __init__(self, input_channels, initial_channels=64):
        super().__init__()
        # Define the first layer increase the num of channels
        self.initial = InitialBlock(input_channels = 3, output_channels = initial_channels)
        # Define the downsampling blocks
        self.downsample = nn.Sequential(
            # Define the first downsampling layer (output n*64*128*128)
            DownBlock(initial_channels, initial_channels,kernel_size = 4, stride = 2,activation="leakyrelu", i_norm = True),
            # Define the second downsampling layer (output n*128*64*64)
            DownBlock(initial_channels, initial_channels * 2, kernel_size = 4, stride = 2,activation="leakyrelu", i_norm = True),
            # Define the third downsampling  layer (output n*256*32*32)
            DownBlock(initial_channels * 2, initial_channels * 4, kernel_size = 4, stride = 2,activation="leakyrelu", i_norm = True),
            # Define the fourth downsampling  layer (output n*512*31*31)
            DownBlock(initial_channels * 4, initial_channels * 8, kernel_size = 4, stride = 1,activation="leakyrelu", i_norm = True),
        )
        # Define the final downsampling  layer
        self.final = nn.Conv2d(initial_channels * 8, 1, kernel_size=4, stride = 1, padding = 1)
            
        
    
    # Define the forward pass
    def forward(self, x):
        
        # Apply the first convolution to increase channels (output n*64*256*256)
        x = self.initial(x)
        # Apply the downsampling layers
        x = self.downsample(x)
        # Apply the final convolution and return the output (output n*1*30*30)
        return torch.sigmoid(self.final(x))
###




# Define the PatchGAN discriminator that uses Depthwise seperable convolutions instead
class CycleGAN_Discriminator_DWS(nn.Module):
    '''
    Discriminator as introduced in CycleGAN paper.
    Discriminator Class. Takes an image as input and outputs a matrix of values classifying if certain patches of image are real or fake.
    Inputs: input_channels -> The number of channels in the input (e.g 3 for RBG image)
            initial_channels -> The number of channels for the first layer of the discriminator
    '''
    
    # Define the constructor
    def __init__(self, input_channels, initial_channels=64):
        super().__init__()
        # Define the first layer increase the num of channels
        self.initial_dws = InitialBlock_DWS(input_channels = 3, output_channels = initial_channels)
        # Define the downsampling blocks
        self.downsample_dws = nn.Sequential(
            # Define the first downsampling layer (output n*64*128*128)
            DownBlock_DWS(initial_channels, initial_channels,kernel_size = 4, stride = 2,activation="leakyrelu", i_norm = True),
            # Define the second downsampling layer (output n*128*64*64)
            DownBlock_DWS(initial_channels, initial_channels * 2, kernel_size = 4, stride = 2,activation="leakyrelu", i_norm = True),
            # Define the third downsampling  layer (output n*256*32*32)
            DownBlock_DWS(initial_channels * 2, initial_channels * 4, kernel_size = 4, stride = 2,activation="leakyrelu", i_norm = True),
            # Define the fourth downsampling  layer (output n*512*31*31)
            DownBlock_DWS(initial_channels * 4, initial_channels * 8, kernel_size = 4, stride = 1,activation="leakyrelu", i_norm = True),
        )
        # Define the final downsampling  layer
        self.final_dws = InitialBlock_DWS(initial_channels * 8, 1, kernel_size=4, stride = 1, padding = 1)
            
        
    
    # Define the forward pass
    def forward(self, x):
        
        # Apply the first convolution to increase channels (output n*64*256*256)
        x = self.initial_dws(x)
        # Apply the downsampling layers
        x = self.downsample_dws(x)
        # Apply the final convolution and return the output (output n*1*30*30)
        return torch.sigmoid(self.final_dws(x))
###