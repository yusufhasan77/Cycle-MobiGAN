### Python file that contains the different loss functions used.

# Import the necessary libraries

# Import pytorch for creating neural networks
import torch
# Import nn module for creating neural networks
import torch.nn as nn


### CycleGAN Loss Functions
# Define the complete generator loss
def gen_loss_complete(real_img_X, real_img_Y,
                      gen_XY, gen_YX, 
                      disc_X, disc_Y,  
                      loss_func_adversarial, loss_func_cycle, loss_func_identity, 
                      lambda_cycle, lambda_identity, 
                      add_identity_loss = True):
    """
    Function to compute the complete generator loss.
    
    Inputs: real_img_X -> The real image from domain X
            real_img_Y -> The real image from domain Y
            disc_X -> Discriminator to classify between real and fake images of domain X
            disc_Y -> Discriminator to classify between real and fake images of domain Y
            genXY -> Generator that maps images from domain X to Y
            genYX -> Generator that maps image from domain Y to X
            loss_func_adversarial -> The loss function used to compute the loss (e.g MSE Loss in CycleGAN)
            loss_func_cycle -> The loss function used to compute the cycle loss (e.g L1 Loss in CycleGAN)
            loss_func_identity -> The loss function used to compute the identity loss (e.g L1 Loss in CycleGAN)
            lambda_cycle -> The weight of cycle consistency loss
            lambda_identity -> The weight of the identity loss
             
    Outputs: The complete generator loss       
    """
    
    # Compute the adversarial loss component for both generators, GenXY and Gen YX
    # Generate a fake image in domain X using a real image in domain Y
    fake_img_X = gen_YX(real_img_Y)
    # Get the discriminator prediction for fake_image_X
    disc_pred_X = disc_X(fake_img_X)
    # Compute and return the discriminator loss for fake image in domain X
    gen_loss_adv_YX = loss_func_adversarial(disc_pred_X, torch.ones_like(disc_pred_X))
    # Generate a fake image in domain Y using a real image in domain X
    fake_img_Y = gen_XY(real_img_X)
    # Get the discriminator prediction for fake_image_Y
    disc_pred_Y = disc_Y(fake_img_Y)
    # Compute and return the discriminator loss for fake image in domain Y
    gen_loss_adv_XY = loss_func_adversarial(disc_pred_Y, torch.ones_like(disc_pred_Y))
    # Add the individually computed losses
    gen_loss_adv = gen_loss_adv_YX + gen_loss_adv_XY
    
    # Compute the cycle consistency loss component for both generators
    gen_cycle_loss_X = loss_func_cycle(real_img_X, gen_YX(fake_img_Y))
    gen_cycle_loss_Y = loss_func_cycle(real_img_Y, gen_XY(fake_img_X))
    gen_loss_cyc = gen_cycle_loss_X + gen_cycle_loss_Y
    
    # Compute the generator loss
    gen_loss = gen_loss_adv + (lambda_cycle * gen_loss_cyc)
    
    # Check if identity loss is to be added
    if add_identity_loss == True:
        # Compute the identity loss
        gen_loss_iden = loss_func_identity(real_img_X, gen_YX(real_img_X)) +  loss_func_identity(real_img_Y, gen_XY(real_img_Y))
        # Add to the generator loss
        gen_loss = gen_loss + (gen_loss_iden * lambda_identity)
        # Return the generator loss
        return gen_loss
    # If identity loss is not to be added, return generator loss
    else:
        return gen_loss
###

# Define the adversarial discriminator loss
def disc_loss(real_img_X, fake_img_X, disc_X, loss_func):
    """
    Function to compute discriminator loss.
    
    Inputs: real_img_X -> The real image of domain X
            fake_img_X -> The image generated in domain X by the generator YX (Y->X) using image in domain Y
            disc_X -> The discriminator to classify between real and fake images of domain X
            loss_func-> The loss function used to compute the loss (MSE Loss in CycleGAN)
            
    Outputs: disc_loss -> The discriminator loss
    """
    
    # Get discriminator prediction for real image
    disc_pred_real = disc_X(real_img_X)
    # Get discriminator prediction for fake image
    disc_pred_fake = disc_X(fake_img_X)
    # Compute and return the discriminator loss
    return (loss_func(disc_pred_real, torch.ones_like(disc_pred_real)) + loss_func(disc_pred_fake, torch.zeros_like(disc_pred_fake)) )
###

### WGAN Loss Functions
### Define the critic loss function

# Define the complete critic loss function
def critic_loss(real_img_X, fake_img_X, real_img_Y, critic_X, gen_YX, device, grad_penalty_weight, BATCH_SIZE = 1):
    """
    Function to compute critic loss.
    
    Inputs: real_img_X -> The real image of domain X
            fake_img_X -> The image generated in domain X by the generator YX (Y->X) using image in domain Y
            critic_X -> The critic to classify between real and fake images of domain X
            loss_func-> The loss function used to compute the loss (MSE Loss in CycleGAN)
            grad_penalty_weight -> The weight of the gradient penalty
            
    Outputs: critic_loss -> The critic loss
    """
    
    # Get critic prediction for real image
    critic_pred_real = critic_X(real_img_X).mean()
    # Get critic prediction for fake image
    critic_pred_fake = critic_X(fake_img_X).mean()

    # Compute the gradient penalty
    # Define the epsilon as the same shape as the images
    epsilon = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, real_img_X.shape[1], 
                                                       real_img_X.shape[2], 
                                                       real_img_X.shape[3]).to(device).requires_grad_()
    # Generate the fake image
    fake_img_X = gen_YX(real_img_Y)
    # Obtain the mixed images
    mixed_images = (real_img_X * epsilon) + (fake_img_X * (1 - epsilon))
    # Compute the critic score on mixed images
    mixed_scores = critic_X(mixed_images)

    # Compute the gradient of the scores with respect to images
    gradient = torch.autograd.grad(
        # Inputs are mixed images
        inputs = mixed_images,
        # Outputs are mixed scores
        outputs = mixed_scores,
        grad_outputs = torch.ones_like(mixed_scores), 
        create_graph = True,
        retain_graph = True,
    )[0]
    
    # Flatten the gradient so each row corresponds to gradient of an image
    gradient = gradient.view(gradient.shape[0], -1)
    
    # Calculate the (L2) norm of every row
    gradient_norm = gradient.norm(2, dim=1)
    
    # Compute and return the gradient penalty
    # Gradient penalty is the mean squared distance of gradient norm and 1
    grad_penalty = mean( (gradient_norm - torch.ones_like(gradient_norm)) ** 2)
     
    # Compute and return the critic loss
    return ( -critic_pred_real + critic_pred_fake + (grad_penalty * grad_penalty_weight) )
###