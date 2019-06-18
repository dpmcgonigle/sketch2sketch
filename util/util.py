"""This module contains simple helper functions """
from __future__ import print_function
import torch
import cv2
import numpy as np
from PIL import Image
import os
from datetime import datetime

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_sample(visuals, basename, img_dir, epoch, total_iters):
    """Save sample images from model at frequency determined by command-line args
    
    Parameters:
        visuals (OrderedDict) - - dictionary of images to display or save
        epoch (int) - - the current epoch
    """
    for label, image in visuals.items():
        image_numpy = tensor2im(image)
        img_path = os.path.join(img_dir, 'epoch_%.3d_iter_%.6d_%s_%s.png' % (epoch, total_iters, label, basename))
        save_image(image_numpy, img_path)

def save_image(image_numpy, image_path):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

#
#   McGonigle datetime stamp in YYYYMMDD_HHMM format
#
def date_time_stamp():
    """returns datetime stamp in YYYYMMDD_HHMM format"""
    now = datetime.now() # current date and time
    return now.strftime("%Y%m%d_%H%M")
    
#
#   McGonigle created function to perform Canny edge detection on pytorch tensor
#
def TorchCanny(x):
    """
    returns a torch tensor Canny edge detection on a torch tensor input.
    input shape should be (num_C x H x W)
    """
        
    # Convert to Numpy
    img = x.numpy()
    dtype = img.dtype

    # Transpose from (c, h, w) to (h, w, c)
    img = img.transpose(1,2,0)
    
    # Convert from (-1.0, 1.0) to (0, 255)
    img = ((img+1) / 2 * 255).round().astype(np.uint8)
        
    # Canny -> BGR2RGB -> bitwise_not
    img = cv2.bitwise_not (
        cv2.cvtColor (
            cv2.Canny (img , 100 , 200),
            cv2.COLOR_BGR2RGB
        )
    )
    
    # Convert from (0, 255) to (-1.0, 1.0)
    img = ((img / 255.0 * 2) - 1).round().astype(dtype)
    
    # Transpose from (h, w, c) to (c, h, w)
    img = img.transpose(2,0,1)
        
    # Put it back into the Torch Tensor
    return torch.FloatTensor(img)
    
#
#   McGonigle used the following function to test the output of the above TorchCanny function
#
def DisplayTorchImg(x):
    """
    use plt.imshow to display a torch image
    This was produced in jupyter notebook to test the TorchCanny method
    """
                
    # Convert to Numpy
    img = x.numpy()

    # Transpose from (c, h, w) to (h, w, c)
    img = img.transpose(1,2,0)

    # Convert from (-1.0, 1.0) to (0, 255)
    img = ((img+1) / 2 * 255).round().astype(np.uint8)

    # Canny -> BGR2RGB -> bitwise_not
    img = cv2.bitwise_not (
        cv2.cvtColor (
            cv2.Canny (img , 100 , 200),
            cv2.COLOR_BGR2RGB
        )
    )
    
    # Display image
    plt.imshow(img)
    
    
def get_img_dir(opt):
    return os.path.join(opt.checkpoints_dir, opt.name, "images")