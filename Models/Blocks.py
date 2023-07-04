### Python file that contains the different blocks used in making the generator, e.g Downsampling, Residual, etc.

# Import the necessary libraries

# Import pytorch for creating neural networks
import torch
# Import nn module for creating neural networks
import torch.nn as nn



### Define the downsampling block
class DownBlock(nn.Module):
    """
    Downblock class. Performs a convolution operation
    
    Inputs: input_channels -> The number of channels in the input (e.g 3 for RBG image)
            kernel_size -> Size of the convolution kernel/filter
            stride -> Stride of the convolution operations
            activation -> Which activation function to use (relu / leakyrelu)
            i_norm -> Whether to use instance normalization (True=yes, False=no)
            
    """
    
    # Define the constructor
    def __init__(self, input_channels, output_channels = 64, kernel_size = 4, stride = 2, activation="relu", i_norm = True):
        super().__init__()
        # Define a convolution layer, the padding is always one in the CycleGAN paper
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size = kernel_size, padding = 1, stride = stride, padding_mode = "reflect")
        # Apply the specified activation
        self.activation = nn.ReLU() if activation == 'relu' else nn.LeakyReLU(0.2)
        # Apply instance norm if it is selected
        if i_norm:
            self.instancenorm = nn.InstanceNorm2d(output_channels) 
        # Set the i_norm attribute to what is provided
        self.i_norm = i_norm
        
    # Define the forward pass
    def forward(self, x):
        # Apply the convolution operator to input x
        x = self.conv(x)
        # Apply the instance norm if required
        if self.i_norm:
            x = self.instancenorm(x)
        # Apply the activation
        x = self.activation(x)
        # Return the output
        return x  
### 



### Define a block that increase number of channels
# Create a convolution block for mapping an input channels to desired output channels
class InitialBlock(nn.Module):
    """
    Initialblock class. Performs a convolution operation to map from input channels to desired channels
    
    Inputs: input_channels -> The number of channels in the input (e.g 3 for RBG image)
            output_channels -> The number of channels desired in the output
    """
    
    # Define the constructor
    def __init__(self, input_channels, output_channels, kernel_size = 7, padding = 3, stride = 1):
        super().__init__()
        # Define a convolution layer, the padding is  three in the CycleGAN paper
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size = kernel_size, padding = padding, stride = stride, padding_mode = "reflect")
        
    # Define the forward pass
    def forward(self, x):
        # Apply the convolution operator to input x and return it
        return self.conv(x)   
###



### Define an upsampling block
class UpBlock(nn.Module):
    """
    Upblock class. Performs a transpose convolution operation
    
    Inputs: input_channels -> The number of channels in the input (e.g 128)
            kernel_size -> Size of the convolution kernel/filter
            stride -> Stride of the convolution operations
            activation -> Which activation function to use (relu / leakyrelu)
            i_norm -> Whether to use instance normalization (True=yes, False=no)
            
    """
    
    # Define the constructor
    def __init__(self, input_channels, output_channels = 64, kernel_size = 3, stride = 2 , activation="relu", i_norm = True):
        super().__init__()
        # Define a transpose convolution layer, the padding is always one 
        self.conv = nn.ConvTranspose2d(input_channels, output_channels, kernel_size = kernel_size, padding = 1, output_padding = 1, stride = stride)
        # Apply the specified activation
        self.activation = nn.ReLU() if activation == 'relu' else nn.LeakyReLU(0.2)
        # Apply instance norm if it is selected
        if i_norm:
            self.instancenorm = nn.InstanceNorm2d(input_channels * 2) 
        # Set the i_norm attribute to what is provided
        self.i_norm = i_norm
        
    # Define the forward pass
    def forward(self, x):
        # Apply the transpose convolution operator to input x
        x = self.conv(x)
        # Apply the instance norm if required
        if self.i_norm:
            x = self.instancenorm(x)
        # Apply the activation
        x = self.activation(x)
        # Return the output
        return x   
###



### Define a residual block
class ResidualBlock(nn.Module):
    """
    ResidualBlock class. Performs a convolution followed by instancenorm then relu.
    After that performs another convolution followed by instance norm.
    Finally input is added to the convolved output.
    
    Inputs: input_channels -> The number of channels in the input (e.g 128)      
    """
    
    # Define the class constructor
    def __init__(self, input_channels):
        super().__init__()
        
        self.conv = nn.Sequential(
            # Define the first convolution
            nn.Conv2d(input_channels, input_channels, kernel_size = 3, padding = 1, stride = 1, padding_mode = "reflect"),
            # Apply the instance norm
            nn.InstanceNorm2d(input_channels),
            # Apply the activation
            nn.ReLU(),
            # Define the second convolution
            nn.Conv2d(input_channels, input_channels, kernel_size = 3, padding = 1, stride = 1, padding_mode = "reflect"),
            # Apply the instance norm
            nn.InstanceNorm2d(input_channels)
        )
        
    # Define the forward pass
    def forward(self, x):
        # Create a copy of the input x
        orig_x = x.clone()
        # Apply the first conv
        x = self.conv(x)
        # Add the original input to the convolved output and return
        return orig_x + x
###



### Define an inverted residual block, it replaces the normal residual block in CycleGAN with inverted residual blocks as in MobileNetv2
class InvertedResidualBlock(nn.Module):
    """
    Inverted ResidualBlock class. Performs a convolution to expand channels followed by instance norm then relu.
    After that performs a depthwise convolution followed by instance norm and relu.
    After that performs a pointwise convolution followed by instance norm and relu.
    Finally input is added to the convolved output.
    
    Inputs: input_channels -> The number of channels in the input (e.g 128)
            expansion_factor -> The factor by which input channels are to be increased
            output_channels -> The number of channels after applying this residual block
    """
    
    # Define the class constructor
    def __init__(self, input_channels, expansion_factor, output_channels):
        super().__init__()
        
        # Define the expansion phase
        self.expansion = nn.Sequential(
            # Apply the convolution to expand the channels
            nn.Conv2d(input_channels, input_channels * expansion_factor, kernel_size = 1, padding = 0, stride = 1, padding_mode = "reflect"),
            # Apply instance norm
            nn.InstanceNorm2d(input_channels * expansion_factor),
            # Apply activation
            nn.ReLU()
        )
        
        # Define the depthwise phase
        self.depthwise = nn.Sequential(
            # Perform a depthwise convolution
            nn.Conv2d(input_channels * expansion_factor, input_channels * expansion_factor, kernel_size = 3, padding = 1, stride = 1, 
                      groups = input_channels * expansion_factor, 
                      padding_mode = "reflect"),
            # Apply instance norm
            nn.InstanceNorm2d(input_channels * expansion_factor),
            # Apply activation
            nn.ReLU()
        )
        
        # Define the pointwise phase
        self.pointwise = nn.Sequential(
            # Apply the pointwise convolution
            nn.Conv2d(input_channels * expansion_factor, output_channels, kernel_size = 1, padding = 0, stride = 1, padding_mode = "reflect"),
            # Apply instance norm
            nn.InstanceNorm2d(output_channels),
            # Apply activation
            nn.ReLU()
        )
        
    # Define the forward pass
    def forward(self, x):
        # Create a copy of the input x
        orig_x = x.clone()
        # Apply the expansion
        x = self.expansion(x)
        # Apply the depthwise
        x = self.depthwise(x)
        # Apply the pointwise
        x = self.pointwise(x)
        
        # Add the original input to the convolved output and return
        return orig_x + x
###

### Define the fully depthwise seperable blocks for the critic

### Define a block that increase number of channels
# Create a depthwise convolution block for mapping an input channels to desired output channels
class InitialBlock_DWS(nn.Module):
    """
    Initialblock class. Performs a depthwise separable convolution operation to map from input channels to desired channels
    
    Inputs: input_channels -> The number of channels in the input (e.g 3 for RBG image)
            output_channels -> The number of channels desired in the output
    """
    
    # Define the constructor
    def __init__(self, input_channels, output_channels, kernel_size = 7, padding = 3, stride = 1):
        super().__init__()
        # Define a depthwise separable convolution layer
        self.conv_dws = nn.Sequential( 
            # Perform the depthwise convolution
            nn.Conv2d(input_channels, input_channels, 
                      kernel_size = kernel_size, padding = padding,
                      stride = stride, padding_mode = "reflect",
                      groups = input_channels),
            # Apply the pointwise convolution
            nn.Conv2d(input_channels, output_channels,
                      kernel_size = 1, padding = 0, stride = 1,
                      padding_mode = "reflect") )
                                      
        
    # Define the forward pass
    def forward(self, x):
        # Apply the convolution operator to input x and return it
        return self.conv_dws(x)   
###

### Define a down block that uses depth convolutions instead of normal convolutions
class DownBlock_DWS(nn.Module):
    """
    Downblock class. Performs a depthwise separable convolution, does instance norm and finally applies activation
    
    Inputs: input_channels -> The number of channels in the input (e.g 3 for RBG image)
            kernel_size -> Size of the convolution kernel/filter
            stride -> Stride of the convolution operations
            activation -> Which activation function to use (relu / leakyrelu)
            i_norm -> Whether to use instance normalization (True=yes, False=no)      
    """
    
    # Define the constructor
    def __init__(self, input_channels, output_channels = 64, kernel_size = 4, stride = 2, activation="relu", i_norm = True):
        super().__init__()
        # Define a convolution layer, the padding is always one in the CycleGAN paper
        self.conv_dws = nn.Sequential( 
            # Perform the depthwise convolution
            nn.Conv2d(input_channels, input_channels, 
                      kernel_size = kernel_size, padding = 1,
                      stride = stride, padding_mode = "reflect",
                      groups = input_channels),
            # Apply the pointwise convolution
            nn.Conv2d(input_channels, output_channels,
                      kernel_size = 1, padding = 0, stride = 1,
                      padding_mode = "reflect") )
        # Apply the specified activation
        self.activation = nn.ReLU() if activation == 'relu' else nn.LeakyReLU(0.2)
        # Apply instance norm if it is selected
        if i_norm:
            self.instancenorm = nn.InstanceNorm2d(output_channels) 
        # Set the i_norm attribute to what is provided
        self.i_norm = i_norm
        
    # Define the forward pass
    def forward(self, x):
        # Apply the convolution operator to input x
        x = self.conv_dws(x)
        # Apply the instance norm if required
        if self.i_norm:
            x = self.instancenorm(x)
        # Apply the activation
        x = self.activation(x)
        # Return the output
        return x  
###

### Define an depthwise upsampling block for the generator
class UpBlock_DWS(nn.Module):
    """
    Upblock class. Performs a transpose depthwise convolution operation
    
    Inputs: input_channels -> The number of channels in the input (e.g 128)
            kernel_size -> Size of the convolution kernel/filter
            stride -> Stride of the convolution operations
            activation -> Which activation function to use (relu / leakyrelu)
            i_norm -> Whether to use instance normalization (True=yes, False=no)
            
    """
    
    # Define the constructor
    def __init__(self, input_channels, output_channels = 64, kernel_size = 3, stride = 2 , activation="relu", i_norm = True):
        super().__init__()
        # Define a transpose convolution layer, the padding is always one 
        self.conv_dws = nn.Sequential( 
            # Perform the depthwise convolution
            nn.ConvTranspose2d(input_channels, input_channels, 
                               kernel_size = kernel_size, 
                               padding = 1, output_padding = 1,
                               stride = stride, groups = input_channels),
            # Apply the pointwise convolution
            nn.Conv2d(input_channels, output_channels,
                      kernel_size = 1, padding = 0, stride = 1,
                      padding_mode = "reflect") )
        # Apply the specified activation
        self.activation = nn.ReLU() if activation == 'relu' else nn.LeakyReLU(0.2)
        # Apply instance norm if it is selected
        if i_norm:
            self.instancenorm = nn.InstanceNorm2d(input_channels * 2) 
        # Set the i_norm attribute to what is provided
        self.i_norm = i_norm
        
    # Define the forward pass
    def forward(self, x):
        # Apply the transpose convolution operator to input x
        x = self.conv_dws(x)
        # Apply the instance norm if required
        if self.i_norm:
            x = self.instancenorm(x)
        # Apply the activation
        x = self.activation(x)
        # Return the output
        return x   
###

### Define an inverted residual block, it replaces the normal residual block in CycleGAN with inverted residual blocks as in MobileNetv2
class ResidualBlock_DWS(nn.Module):
    """
    Depthwise Seperable Residual Block class. Same as residual block but convolution is depthwise.
    
    Inputs: input_channels -> The number of channels in the input (e.g 128)
            output_channels -> The number of channels after applying this depthwise separable residual block
    """
    
    # Define the class constructor
    def __init__(self, input_channels, output_channels):
        super().__init__()
        
        # Define the depthwise phase
        self.depthwise = nn.Sequential(
            # Perform a depthwise convolution
            nn.Conv2d(input_channels, input_channels, kernel_size = 3, padding = 1, stride = 1, 
                      groups = input_channels, 
                      padding_mode = "reflect"),
            # Apply instance norm
            nn.InstanceNorm2d(input_channels),
            # Apply activation
            nn.ReLU()
        )
        
        # Define the pointwise phase
        self.pointwise = nn.Sequential(
            # Apply the pointwise convolution
            nn.Conv2d(input_channels, output_channels, kernel_size = 1, padding = 0, stride = 1, padding_mode = "reflect"),
            # Apply instance norm
            nn.InstanceNorm2d(output_channels),
            # Apply activation
            nn.ReLU()
        )
        
    # Define the forward pass
    def forward(self, x):
        # Create a copy of the input x
        orig_x = x.clone()
        # Apply the depthwise
        x = self.depthwise(x)
        # Apply the pointwise
        x = self.pointwise(x)
        
        # Add the original input to the convolved output and return
        return orig_x + x
###
