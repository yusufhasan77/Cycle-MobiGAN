### Python file that contains the different types of Generators used CycleGAN, CycleGAN_DW, etc.

# Import the necessary libraries
# Import pytorch for creating neural networks
import torch
# Import nn module for creating neural networks
import torch.nn as nn
# Import the blocks used to create the discriminator
from .Blocks import *



# Define the CycleGAN generator
class CycleGAN_Generator(nn.Module):
    '''
    Generator architecture as implemented in CycleGAN paper
    Generator Class. Takes an image of class A as input and outputs an image of class B
    Inputs: input_channels -> The number of channels in the input (e.g 3 for RBG image)
            initial_channels -> The number of channels for the first layer of the generator
            output_channels -> The number of channels expected in the output (3 for RGB image)
    '''
    
    # Define the class constructor
    def __init__(self, input_channels = 3, initial_channels = 64, output_channels = 3):
        super().__init__()
        # Define the first to increase the number of channels
        self.initial = InitialBlock(input_channels = input_channels, output_channels = initial_channels, kernel_size = 7, padding = 3, stride = 1)
        
        # Define the downsampling layers
        self.down = nn.Sequential(
            # Define the first downsampling layer
            DownBlock(initial_channels, initial_channels * 2, kernel_size = 3, stride = 2,activation="relu", i_norm = True),
            # Define the second downsampling layer
            DownBlock(initial_channels * 2, initial_channels * 4, kernel_size = 3, stride = 2,activation="relu", i_norm = True) )
        
        # Define the residual blocks
        # Create a list of the residual blocks as they are identical
        self.res = nn.ModuleList( [ResidualBlock(initial_channels * 4) for layers in range(1, 10)] )
        # Create a neural network from the list of residual blocks
        self.res = nn.Sequential(*self.res)
        
        # Define the upscaling layers
        self.up = nn.Sequential(
            # Define the first upsampling block
            UpBlock(initial_channels * 4, output_channels = initial_channels * 2, kernel_size = 3, 
                           stride = 2 , activation="relu", i_norm = True),
            # Define the second upsampling block
            UpBlock(initial_channels * 2, output_channels = initial_channels, kernel_size = 3, 
                           stride = 2 , activation="relu", i_norm = True) )

        # Define the final layer to decrease the number of channels to 3
        self.final = InitialBlock(input_channels = initial_channels, output_channels = output_channels, kernel_size = 7, padding = 3, stride = 1)
        
        # Define the activation
        self.tanh = torch.nn.Tanh()
        
    # Define the forward pass
    def forward(self, x):
        # Increase the number of channels
        x = self.initial(x)
        # Downsampling layers
        x = self.down(x)
        # residual blocks
        x = self.res(x)
        # Upsampling layers
        x = self.up(x)
        # Reduce the number of channels to 3
        x = self.final(x)
        # Apply activation and return the output
        return self.tanh(x)
###



# Define the CycleGAN generator
class CycleGAN_Generator_DWS(nn.Module):
    '''
    Generator architecture similar the one presented in CycleGAN paper, 
    however all convolution layers have been replaced by depthwise separable convolutions
    Generator Class. Takes an image of class X as input and outputs an image of class Y
    Inputs: input_channels -> The number of channels in the input (e.g 3 for RBG image)
            initial_channels -> The number of channels for the first layer of the generator
            output_channels -> The number of channels expected in the output (3 for RGB image)
    '''
    
    # Define the class constructor
    def __init__(self, input_channels = 3, initial_channels = 64, output_channels = 3, res_blocks = 9):
        super().__init__()
        # Create an attribute that stores default res blocks
        self.res_blocks = res_blocks
        # Define the first to increase the number of channels
        self.initial_dws = InitialBlock_DWS(input_channels = input_channels, output_channels = initial_channels, kernel_size = 7, padding = 3, stride = 1)
        
        # Define the downsampling layers
        self.down_dws = nn.Sequential(
            # Define the first downsampling layer
            DownBlock_DWS(initial_channels, initial_channels * 2, kernel_size = 3, stride = 2,activation="relu", i_norm = True),
            # Define the second downsampling layer
            DownBlock_DWS(initial_channels * 2, initial_channels * 4, kernel_size = 3, stride = 2,activation="relu", i_norm = True) )
        
        # Define the residual blocks
        # Create a list of the residual blocks as they are identical
        self.res_dws = nn.ModuleList( [ResidualBlock_DWS(input_channels = initial_channels * 4, output_channels = initial_channels * 4) 
                                       for layers in range(1, self.res_blocks + 1)] )
        # Create a neural network from the list of residual blocks
        self.res_dws = nn.Sequential(*self.res_dws)
        
        # Define the upscaling layers
        self.up_dws = nn.Sequential(
            # Define the first upsampling block
            UpBlock_DWS(initial_channels * 4, output_channels = initial_channels * 2, kernel_size = 3, 
                           stride = 2 , activation="relu", i_norm = True),
            # Define the second upsampling block
            UpBlock_DWS(initial_channels * 2, output_channels = initial_channels, kernel_size = 3, 
                           stride = 2 , activation="relu", i_norm = True) )

        # Define the final layer to decrease the number of channels to 3
        self.final_dws = InitialBlock_DWS(input_channels = initial_channels, output_channels = output_channels, kernel_size = 7, padding = 3, stride = 1)
        
        # Define the activation
        self.tanh = torch.nn.Tanh()
        
    # Define the forward pass
    def forward(self, x):
        # Increase the number of channels
        x = self.initial_dws(x)
        # Downsampling layers
        x = self.down_dws(x)
        # residual blocks
        x = self.res_dws(x)
        # Upsampling layers
        x = self.up_dws(x)
        # Reduce the number of channels to 3
        x = self.final_dws(x)
        # Apply activation and return the output
        return self.tanh(x)
###



# Define the CycleGAN generator with inverted residual blocks
class CycleGAN_IR_Generator(nn.Module):
    '''
    Generator architecture similar to the one implemented in CycleGAN paper,
    residual blocks have been replaced by inverted residual blocks as in mobilenetv2
    Generator Class. Takes an image of class A as input and outputs an image of class B
    Inputs: input_channels -> The number of channels in the input (e.g 3 for RBG image)
            initial_channels -> The number of channels for the first layer of the generator
            output_channels -> The number of channels expected in the output (3 for RGB image)
    '''
    
    # Define the class constructor
    def __init__(self, input_channels = 3, initial_channels = 64, output_channels = 3, res_blocks = 9, expansion_factor = 6):
        super().__init__()
        # Create an attribute that stores the amount of res blocks
        self.res_blocks = res_blocks
        # Create an attribute to store the amount of expansion factor needed
        self.expansion_factor = expansion_factor
        
        # Define the first to increase the number of channels
        self.initial = InitialBlock(input_channels = input_channels, output_channels = initial_channels, kernel_size = 7, padding = 3, stride = 1)
        
        # Define the downsampling layers
        self.down = nn.Sequential(
            # Define the first downsampling layer
            DownBlock(initial_channels, initial_channels * 2, kernel_size = 3, stride = 2,activation="relu", i_norm = True),
            # Define the second downsampling layer
            DownBlock(initial_channels * 2, initial_channels * 4, kernel_size = 3, stride = 2,activation="relu", i_norm = True) )
        
        # Define the residual blocks
        # Create a list of the residual blocks as they are identical
        self.inverted_res = nn.ModuleList( [InvertedResidualBlock(input_channels = initial_channels * 4, expansion_factor = self.expansion_factor, 
                                                                  output_channels = initial_channels * 4) 
                                            for layers in range(1, self.res_blocks + 1)] )
        # Create a neural network from the list of residual blocks
        self.inverted_res = nn.Sequential(*self.inverted_res)
        
        # Define the upscaling layers
        self.up = nn.Sequential(
            # Define the first upsampling block
            UpBlock(initial_channels * 4, output_channels = initial_channels * 2, kernel_size = 3, 
                           stride = 2 , activation="relu", i_norm = True),
            # Define the second upsampling block
            UpBlock(initial_channels * 2, output_channels = initial_channels, kernel_size = 3, 
                           stride = 2 , activation="relu", i_norm = True) )

        # Define the final layer to decrease the number of channels to 3
        self.final = InitialBlock(input_channels = initial_channels, output_channels = output_channels, kernel_size = 7, padding = 3, stride = 1)
        
        # Define the activation
        self.tanh = torch.nn.Tanh()
        
    # Define the forward pass
    def forward(self, x):
        # Increase the number of channels
        x = self.initial(x)
        # Downsampling layers
        x = self.down(x)
        # residual blocks
        x = self.inverted_res(x)
        # Upsampling layers
        x = self.up(x)
        # Reduce the number of channels to 3
        x = self.final(x)
        # Apply activation and return the output
        return self.tanh(x)
###


# Define the CycleGAN generator
class CycleGAN_IR_Generator_DWS(nn.Module):
    '''
    Generator architecture similar the one presented in CycleGAN paper, 
    however all convolution layers have been replaced by depthwise separable convolutions
    Generator Class. Takes an image of class X as input and outputs an image of class Y
    Inputs: input_channels -> The number of channels in the input (e.g 3 for RBG image)
            initial_channels -> The number of channels for the first layer of the generator
            output_channels -> The number of channels expected in the output (3 for RGB image)
    '''
    
    # Define the class constructor
    def __init__(self, input_channels = 3, initial_channels = 64, output_channels = 3, res_blocks = 9, expansion_factor = 6):
        super().__init__()
        # Create an attribute that stores the amount of res blocks
        self.res_blocks = res_blocks
        # Create an attribute to store the amount of expansion factor needed
        self.expansion_factor = expansion_factor
        # Define the first to increase the number of channels
        self.initial_dws = InitialBlock_DWS(input_channels = input_channels, output_channels = initial_channels, kernel_size = 7, padding = 3, stride = 1)
        
        # Define the downsampling layers
        self.down_dws = nn.Sequential(
            # Define the first downsampling layer
            DownBlock_DWS(initial_channels, initial_channels * 2, kernel_size = 3, stride = 2,activation="relu", i_norm = True),
            # Define the second downsampling layer
            DownBlock_DWS(initial_channels * 2, initial_channels * 4, kernel_size = 3, stride = 2,activation="relu", i_norm = True) )
        
        # Define the residual blocks
        # Create a list of the residual blocks as they are identical
        self.res_dws = nn.ModuleList( [InvertedResidualBlock(input_channels = initial_channels * 4, expansion_factor = self.expansion_factor,
                                                             output_channels = initial_channels * 4) 
                                       for layers in range(1, self.res_blocks + 1)] )
        # Create a neural network from the list of residual blocks
        self.res_dws = nn.Sequential(*self.res_dws)
        
        # Define the upscaling layers
        self.up_dws = nn.Sequential(
            # Define the first upsampling block
            UpBlock_DWS(initial_channels * 4, output_channels = initial_channels * 2, kernel_size = 3, 
                           stride = 2 , activation="relu", i_norm = True),
            # Define the second upsampling block
            UpBlock_DWS(initial_channels * 2, output_channels = initial_channels, kernel_size = 3, 
                           stride = 2 , activation="relu", i_norm = True) )

        # Define the final layer to decrease the number of channels to 3
        self.final_dws = InitialBlock_DWS(input_channels = initial_channels, output_channels = output_channels, kernel_size = 7, padding = 3, stride = 1)
        
        # Define the activation
        self.tanh = torch.nn.Tanh()
        
    # Define the forward pass
    def forward(self, x):
        # Increase the number of channels
        x = self.initial_dws(x)
        # Downsampling layers
        x = self.down_dws(x)
        # residual blocks
        x = self.res_dws(x)
        # Upsampling layers
        x = self.up_dws(x)
        # Reduce the number of channels to 3
        x = self.final_dws(x)
        # Apply activation and return the output
        return self.tanh(x)
###



### Define a CycleGAN Generator which completely utilizes depthwise seperable convolutions
# Define the CycleGAN generator with inverted residual blocks
class CycleWGAN_GP_DWS_Generator(nn.Module):
    '''
    Custom generator architecture. Similar to the one introduced in CycleGAN paper. 
    The residual blocks have been replaced with inverted residual blocks,
    and all convolution layers have been replaced with depthwise seperable convolutions.
    Generator Class. Takes an image of class A as input and outputs an image of class B
    Inputs: input_channels -> The number of channels in the input (e.g 3 for RBG image)
            initial_channels -> The number of channels for the first layer of the generator
            output_channels -> The number of channels expected in the output (3 for RGB image)
    '''
    
    # Define the class constructor
    def __init__(self, input_channels = 3, initial_channels = 64, output_channels = 3, expansion_factor = 2):
        super().__init__()
        # Define the first to increase the number of channels
        self.initial_dws = InitialBlock_DWS(input_channels = input_channels, output_channels = initial_channels, kernel_size = 7, padding = 3, stride = 1)
        
        # Define the downsampling layers
        self.down_dws = nn.Sequential(
            # Define the first downsampling layer
            DownBlock_DWS(initial_channels, initial_channels * 2, kernel_size = 3, stride = 2,activation="relu", i_norm = True),
            # Define the second downsampling layer
            DownBlock_DWS(initial_channels * 2, initial_channels * 4, kernel_size = 3, stride = 2,activation="relu", i_norm = True) )
        
        # Define the residual blocks
        # Create a list of the residual blocks as they are identical
        self.inverted_res = nn.ModuleList( [InvertedResidualBlock(input_channels = initial_channels * 4, expansion_factor = expansion_factor, 
                                                                  output_channels = initial_channels * 4) 
                                            for layers in range(1, 10)] )
        # Create a neural network from the list of residual blocks
        self.inverted_res = nn.Sequential(*self.inverted_res)
        
        # Define the upscaling layers
        self.up_dws = nn.Sequential(
            # Define the first upsampling block
            UpBlock_DWS(initial_channels * 4, output_channels = initial_channels * 2, kernel_size = 3, 
                           stride = 2 , activation="relu", i_norm = True),
            # Define the second upsampling block
            UpBlock_DWS(initial_channels * 2, output_channels = initial_channels, kernel_size = 3, 
                           stride = 2 , activation="relu", i_norm = True) )

        # Define the final layer to decrease the number of channels to 3
        self.final_dws = InitialBlock_DWS(input_channels = initial_channels, output_channels = output_channels, kernel_size = 7, padding = 3, stride = 1)
        
        # Define the activation
        self.tanh = torch.nn.Tanh()
        
    # Define the forward pass
    def forward(self, x):
        # Increase the number of channels
        x = self.initial_dws(x)
        # Downsampling layers
        x = self.down_dws(x)
        # residual blocks
        x = self.inverted_res(x)
        # Upsampling layers
        x = self.up_dws(x)
        # Reduce the number of channels to 3
        x = self.final_dws(x)
        # Apply activation and return the output
        return self.tanh(x)
###