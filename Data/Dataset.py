### Python file that contains the class to load the data.

# Import the necessary libraries
# Import the os for working with directories and paths
import os
# Import randint as this will be used to make sure images are not paired
from random import randint
# Import Image from PIL to load images
from PIL import Image
# Import the torch Dataset class to create a Dataset class
from torch.utils.data import Dataset
# Import the transform in case any transformation is to be done
import torchvision.transforms as transforms


# Define a class to load the dataset
class LoadDataset(Dataset):
    
    # Define the constructor
    def __init__(self, img_dir = "apple2orange", mode = "train", transform=None):
        # Define the transform attribute
        self.transform = transform
        # Create an attribute to store the name of image directory
        self.img_dir = img_dir
        # Create an atrribute to store the mode
        self.mode = mode
        # Define the address of images of domain X
        self.path_img_X = sorted(os.listdir( os.path.join(self.img_dir, self.mode + "A") ) )
        # Define the address of images of domain Y
        self.path_img_Y = sorted(os.listdir( os.path.join(self.img_dir, self.mode + "B") ) )
        # Handle the case where path_img_X has more files than path_img_Y, this code is written assuming Y has more files
        if len(self.path_img_X) > len(self.path_img_Y):
            # Switch the paths
            self.path_img_X, self.path_img_Y = self.path_img_Y, self.path_img_X

    # Define the method that returns the length of the dataset
    def __len__(self):
        # Return the length of the dataset
        return min( len(self.path_img_X), len(self.path_img_Y) )

    def __getitem__(self, index_image):
        
        # Check if there is any transformation to be done
        if self.transform != None:
            # Get an image of in domain X
            real_image_X = self.transform( Image.open(
                os.path.join(self.img_dir, self.mode + "A/") +
                self.path_img_X[index_image % len(self.path_img_X)] ).convert("RGB") )
            # Get an image in domain Y, make sure it is random so there are no paired images
            real_image_Y = self.transform( Image.open(
                os.path.join(self.img_dir, self.mode + "B/") +
                self.path_img_Y[randint(0, len(self.path_img_Y) - 1)] ).convert("RGB") )
            # Return the images
            return real_image_X, real_image_Y
        # If the there is no transform just return the images
        else:
            # Get an image in domain X
            real_image_X = Image.open( os.path.join(self.img_dir, self.mode + "A/")
                                          + self.path_img_X[index_image % len(self.path_img_X)] ).convert("RGB")
            # Get an image in domain Y, make sure it is random so there are no paired images
            real_image_Y = Image.open( os.path.join(self.img_dir, self.mode + "B/") 
                                          + self.path_img_Y[randint(0, len(self.path_img_Y) - 1)] ).convert("RGB")
            # Return the images
            return real_image_X, real_image_Y
###