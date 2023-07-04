### Python file that contains the visualization function used.

# Import the necessary libraries
# Import pytorch for creating neural networks
import torch
# Import nn module for creating neural networks
import torch.nn as nn
# Import matplotlib pyplot for displaying images
import matplotlib.pyplot as plt

# Define a function to visualize the images
def visualize_images(real_img_X, real_img_Y, fake_img_X, fake_img_Y):
    """
    Function to visualize the real and generated images.
    
    Inputs: real_img_X -> The real image in the domain X
            real_img_Y -> The real image in the domain Y
            fake_img_X -> The generated image in the domain X using 
                          a real image in domain Y and gen(YX)
            fake_img_Y -> The generated image in the domain Y using 
                          a real image in domain X and gen(XY)
    
    Outputs: Displays the real and generated images
    """
    # Setup a 2x2 subplot
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10,10))
    
    # Display the real_img_X (real horse) in top left
    ax[0, 0].imshow( (real_img_X.detach().cpu().squeeze().permute(1,2,0)))
    # Display the real_img_Y (real zebra) in top right
    ax[0, 1].imshow( (real_img_Y.detach().cpu().squeeze().permute(1,2,0)))
    # Display the fake_img_Y (fake horse converted to zebra) in bottom left
    ax[1, 0].imshow( (fake_img_Y.detach().cpu().squeeze().permute(1,2,0)))
    # Display the fake_img_X (fake zebra converted to horse) in bottom right
    ax[1, 1].imshow( (fake_img_X.detach().cpu().squeeze().permute(1,2,0)))
    # Make sure the subplots are tightly packed
    plt.tight_layout()
    # Show the plots
    plt.show()