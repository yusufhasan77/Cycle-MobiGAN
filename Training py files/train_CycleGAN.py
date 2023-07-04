#!/usr/bin/env python
# coding: utf-8

# # CycleGAN

# ## Libraries

# Import the necessary libraries
# Import pytorch for creating neural networks
import torch
# Import nn module for creating neural networks
import torch.nn as nn
# import dataloader for loading data
from torch.utils.data import DataLoader
# Import transforms
from torchvision import transforms


# Import tqdm
from tqdm import tqdm
# Import torch summary to get a summary of the model
#from torchsummary import summary


# Import the functions for training the models
# Import the discriminators
from Models.Discriminators import *
# Import the generators
from Models.Generators import *
# Import the loss functions
from Models.Loss_functions import disc_loss, gen_loss_complete

# Import the function to load the data
from Data.Dataset import *


# Import the visualization function
from Utils.visualize_images import visualize_images


# ## Model training parameters

# Set the device
device = "cuda" if torch.cuda.is_available() else "cpu"
# Display the device
print(device)


# Define the model training parameters
# Set the number of epochs
NUM_EPOCHS = 201
# Set the training batch size
BATCH_SIZE = 1
# Set the learning rate to be used for the optimizers
LEARNING_RATE = 0.0002
# The size of the images after being loaded
LOAD_SHAPE = 286
# The size of the images after being cropped
CROPPED_SHAPE = 256
# Set seed
torch.manual_seed(7)


# ## Loading the dataset

# Define a transformation to be applied to images
transform = transforms.Compose([
    transforms.Resize(LOAD_SHAPE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(CROPPED_SHAPE),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])


# Load the dataset
dataset = LoadDataset("Data/apple2orange", transform  = transform, mode="train")


# Create a dataloader
dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle=True)


# ## Function for training the model

# Define a function to initialize the weights
def weights_init(m):
    # Check if the current layer is a Conv2d or ConvTranspose2d
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # Apply weights sampled from a normal distribution
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    # Check if the current layer is a normalization layer
    if isinstance(m, nn.BatchNorm2d):
        # Apply weights sampled from a normal distribution
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        # Set the bias as zero
        torch.nn.init.constant_(m.bias, 0)


# Create the generators
gen_XY = CycleGAN_Generator(3, 64, 3 ).to(device)
gen_YX = CycleGAN_Generator(3, 64, 3 ).to(device)
# Create the optimizer for the generators
gen_opt = torch.optim.Adam(list(gen_XY.parameters()) + list(gen_YX.parameters()), lr=LEARNING_RATE, betas=(0.5, 0.999))

# Create discriminator X and it's optimizer
disc_X = CycleGAN_Discriminator(3, 64).to(device)
disc_X_opt = torch.optim.Adam(disc_X.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
# Create discriminator Y and it's optimizer
disc_Y = CycleGAN_Discriminator(3, 64).to(device)
disc_Y_opt = torch.optim.Adam(disc_Y.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))


# Set whether to use a checkpoint and continue from there or start training from scratch
load_model = False
if load_model == True:
    # Load the model weights and optimizer 
    CycleGAN_dict = torch.load("Model Weights/CycleGAN/CycleGAN_epoch100.pth")
    # Apply the weights and optimizer
    gen_XY.load_state_dict(CycleGAN_dict['gen_XY'])
    gen_YX.load_state_dict(CycleGAN_dict['gen_YX'])
    gen_opt.load_state_dict(CycleGAN_dict['gen_opt'])
    disc_X.load_state_dict(CycleGAN_dict['disc_X'])
    disc_Y.load_state_dict(CycleGAN_dict['disc_Y'])
    disc_X_opt.load_state_dict(CycleGAN_dict['disc_X_opt'])
    disc_Y_opt.load_state_dict(CycleGAN_dict['disc_Y_opt'])
else:
    # Apply the weights to the generators and discriminators sampled from a normal distribution
    gen_XY = gen_XY.apply(weights_init)
    gen_YX = gen_YX.apply(weights_init)
    disc_X = disc_X.apply(weights_init)
    disc_Y = disc_Y.apply(weights_init)


# Define loss functions
loss_func_adversarial = nn.MSELoss()
loss_func_cycle = nn.L1Loss()
loss_func_identity = nn.L1Loss()

# Define the weights for the loss functions
lambda_cycle = 10
lambda_identity = 10


# Print the summary of the Generator XY
#summary(gen_XY, (3, 256, 256))

# Print the summary of the Generator YX
#summary(gen_YX, (3, 256, 256))

# Print the summary of the Discriminator X
#summary(disc_X, (3, 256, 256))

# Print the summary of the Discriminator Y
#summary(disc_Y, (3, 256, 256))


visl = False


# Define the function to train the CycleGAN
def train_CycleGAN():
    """
    Function to train CycleGAN. Doesn't take any inputs.
    
    Outputs: Saves the model to disk
             Displays the real and generated images after every few steps
             Prints the mean generator and discriminator losses
    """
    
    # Define variables to keep track of mean generator and discriminator losses
    loss_generator_mean = 0
    loss_discriminator_mean = 0
    # Create a counter that to keep track of how many images have been processed
    # and to display real and generated images after certain images have been processed
    disp = 1
    
    # Keep training for a certain number of epochs
    for epoch in range(1, NUM_EPOCHS):
        # Use the tqdm to load the data
        for real_img_X, real_img_Y in tqdm(dataloader):
            # Move the images to the device
            real_img_X = real_img_X.to(device)
            real_img_Y = real_img_Y.to(device)
                   
            # Update the generators
            # Set the gradients to zero
            gen_opt.zero_grad()
            # Compute the generator loss
            loss_gen = gen_loss_complete(real_img_X, real_img_Y,
                                         gen_XY, gen_YX, disc_X, disc_Y,  
                                         loss_func_adversarial, loss_func_cycle, loss_func_identity, 
                                         lambda_cycle, lambda_identity, 
                                         add_identity_loss = True)
            # Perform backpropogation for the generator
            loss_gen.backward()
            # Update the generators by taking an optimization step
            gen_opt.step()

            # Update disc_X 
            # Set the gradients to zero
            disc_X_opt.zero_grad()
            # Generate a fake image in domain X
            with torch.no_grad():
                fake_img_X = gen_YX(real_img_Y)
            # Compute the loss
            loss_disc_X = disc_loss(real_img_X, fake_img_X.detach(), disc_X, loss_func_adversarial)
            # Perform backpropogation for disc_X
            loss_disc_X.backward(retain_graph=True)
            # Update disc_X opt
            disc_X_opt.step() 
            
            # Update disc_Y
            # Set the gradients to zero
            disc_Y_opt.zero_grad() 
            # Generate a fake image in domain Y
            with torch.no_grad():
                fake_img_Y = gen_XY(real_img_X)
            # Compute the loss
            loss_disc_Y = disc_loss(real_img_Y, fake_img_Y.detach(), disc_Y, loss_func_adversarial)
            # Perform backpropogation for disc_Y
            loss_disc_Y.backward(retain_graph=True)
            # Update disc_Y opt
            disc_Y_opt.step() 
            
            # Compute the mean generator and discriminator loss
            loss_generator_mean = loss_gen.item()/disp
            loss_discriminator_mean = ( loss_disc_X + loss_disc_Y ).item() / disp
            
            # Check if the images are to be visualized
            if visl == True:
                # Code for visualization
                if disp % 10 == 0:
                    # Visualize the images
                    # The transform applied to the images 0.5 and divides by 0.5
                    # To un-transform we have to add 1 and divide by 2
                    visualize_images( (real_img_X +1 ) / 2, (real_img_Y + 1) / 2, (fake_img_X + 1) / 2, (fake_img_Y + 1) / 2)
                    # Display the mean generator and discriminator losses
                    print(f"Display step is: {disp}\n"
                      f"The average discriminator loss is: {loss_discriminator_mean}\n"
                      f"The average generator loss is: {loss_generator_mean}\n")
            # Increase the display counter
            disp+= 1
        # For loop for loading data using tqdm ends here
        
        # Print the generator and discriminator loss at each epoch
        print(f"Epoch is: {epoch}\n"
              f"The average discriminator loss is: {loss_discriminator_mean}\n"
              f"The average generator loss is: {loss_generator_mean}\n")
            
        # Save the model after every 10 epochs
        if epoch % 10 == 0:
            # Define a save path
            save_path = "Model Weights/CycleGAN/"
            # Create a variable storing the current checkpoint name
            name_model_epoch = f"CycleGAN_epoch{epoch}.pth"
            # Create a directory if it does not exist
            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok = True)
            # Save the model to this directory
            torch.save({
                    'gen_XY': gen_XY.state_dict(),
                    'gen_YX': gen_YX.state_dict(),
                    'gen_opt': gen_opt.state_dict(),
                    'disc_X': disc_X.state_dict(),
                    'disc_Y': disc_Y.state_dict(),
                    'disc_X_opt':disc_X_opt.state_dict(),
                    'disc_Y_opt':disc_Y_opt.state_dict()
                }, save_path + name_model_epoch)
# End of function definition        


# Train the CycleGAN
train_CycleGAN()
