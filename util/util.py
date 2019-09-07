"""This module contains simple helper functions """
from __future__ import print_function
import torch
import cv2
import numpy as np
from PIL import Image
import os
from datetime import datetime
import matplotlib.pyplot as plt

############################################################################################
#
#   U   T   I   L   I   T   Y           F   U   N   C   T   I   O   N   S
#   
#   tensor2im                           converts a tensor array into numpy image array
#   diagnose_networks                   Calculate and print the mean of average absolute(gradients)
#   save_sample                         Save sample images from model at frequency with command-line args
#   save_image                          Save a numpy image to the disk
#   print_numpy                         Print the mean, min, max, median, std, and size of a numpy array
#   makedirs                            create empty directories if they don't exist
#   makedir                             create empty directory if it doesn't exist
#   date_time_stamp                     returns datetime stamp in YYYYMMDD_HHMM format
#   TorchCanny                          returns a torch tensor Canny edge detection
#   DisplayTorchImg                     use plt.imshow to display a torch image
#   get_img_dir                         returns image directory for a given experiment and phase
#   plot_loss                           creates a matplotlib loss chart at the end of training
#
############################################################################################
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

############################################################################################

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

############################################################################################

def save_sample(visuals, basename, img_dir, epoch=None, total_iters=None):
    """Save sample images from model at frequency determined by command-line args
    
    Parameters:
        visuals (OrderedDict) - - dictionary of images to display or save
        epoch (int) - - the current epochcreate empty directories if they don't exist
    """
    for label, image in visuals.items():
        image_numpy = tensor2im(image)
        
        # Save image without epoch or total iters
        if epoch is None and total_iters is None:
            img_path = os.path.join(img_dir, '%s_%s.png' % (label, basename))
        else:
            img_path = os.path.join(img_dir, 'epoch_%.3d_iter_%.8d_%s_%s.png' % (epoch, total_iters, label, basename))
        save_image(image_numpy, img_path)

############################################################################################
def save_image(image_numpy, image_path):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

############################################################################################

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

############################################################################################

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

############################################################################################

def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

############################################################################################
#
#   McGonigle datetime stamp in YYYYMMDD_HHMM format
#
def date_time_stamp():
    """returns datetime stamp in YYYYMMDD_HHMM format"""
    now = datetime.now() # current date and time
    return now.strftime("%Y%m%d_%H%M")
    
############################################################################################
#
#   McGonigle created function to perform Canny edge detection on pytorch tensor
#   6/29/2019 - Updated to handle single images and batches, RGB and grayscale
#
def TorchCanny(x):
    """
    returns a torch tensor Canny edge detection on a torch tensor input.  Can be used on batches.
    input can be in shape (num_imgs X num_channels X height X width) or  (num_channels X height X width)
    output will match input shape
    """
    # Expand to include dimension representing number of images for for loop
    dims = x.ndimension()
    if dims == 3:
        x = x.unsqueeze(0)
    
    # Get number of images and channels
    n_imgs = x.shape[0]
    n_channels = x.shape[1]
    
    arr = np.empty_like(x.numpy())
    
    # Loop through images in batch
    for i in range(n_imgs):
        
        # Convert to Numpy
        img = x[i].numpy()
        dtype = img.dtype
        
        # Transpose from (c, h, w) to (h, w, c)
        img = img.transpose(1,2,0)
       
        # Convert from (-1.0, 1.0) to (0, 255)
        img = ((img+1) / 2 * 255).round().astype(np.uint8)
        
        # Canny -> BGR2RGB or keep grayscale -> bitwise_not
        # Edge detection
        img = cv2.Canny (img , 100 , 200)
        
        # Convert to color or keep grayscale depending on n_channels
        if n_channels == 3:
            img = cv2.cvtColor (img, cv2.COLOR_BGR2RGB)
        
        # bitwise not so the background is white and contours are black
        img = cv2.bitwise_not (img)
        
        # Canny and bitwise not will squeeze the single dimension if it is a grayscale
        if n_channels == 1:
            img = np.expand_dims(img, -1)
            
        # Convert from (0, 255) to (-1.0, 1.0)
        img = ((img / 255.0 * 2) - 1).round().astype(dtype)
        
        # Transpose from (h, w, c) to (c, h, w)
        img = img.transpose(2,0,1)
        
        # Add image to arr
        arr[i] = img
    
    if dims == 3:
        arr = arr.squeeze(0)
    
    # Put it back into the Torch Tensor
    return torch.FloatTensor(arr)
    
############################################################################################
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
    
############################################################################################
    
def get_img_dir(opt):
    """ returns image directory based on checkpoints_dir, experiment name, and phase (train, val, test) """
    return os.path.join(opt.checkpoints_dir, opt.name, opt.phase)

############################################################################################

def plot_loss(opt, start=None, end=None):
    """ plots loss_log.txt as a matplotlib graph at the end of training """

    #
    #   defaults for start and end based off command line options if not specified in function call
    #
    if not start:
        start = opt.start_epoch
    if not end:
        end = (opt.niter + opt.niter_decay)

    log_file = os.path.join(opt.expr_dir, "loss_log.txt")
    lossdict = {}               #   store text losses for each model
    epochs = end - start + 1    #   start is typically 1; so start 1, end 20 should yield 20 epochs

    try:

        #
        #   Open the loss_log.txt file and gather all of the loss values into lossdict
        #
        with open (log_file, 'r') as log:
            lines = log.readlines()
            for line in lines[2:]:
                line = line.split(')')[-1]
                words = line.split()
            
                name = None
                for index, word in enumerate(words):
          
                    if index % 2 == 0:
                        name=word[:-1]
                        if name not in lossdict.keys():
                            lossdict[name] = []
                    else:
                        try:
                            lossdict[name].append( float(word[:-1]) )
                        except:
                            pass

            #
            #   Plot a long horizontal graph
            #
            plt.figure(figsize=(24, 8))
            plt.ylabel("Loss")
            plt.xlabel("Epoch")
            for key, value in lossdict.items():
                arr = np.clip( np.array(value) , a_min=-10, a_max=10)
                x = [x * (epochs / arr.shape[0]) for x in range(arr.shape[0])]
                plt.plot(x, arr, label="%s loss"%key)
            plt.legend(loc='upper left')
            plt.savefig(os.path.join(opt.expr_dir, "epoch_%d-%d_losses.png"))

    except Exception as e:
        print("Couldn't plot graph from log file %s. ERROR: %s "%(log_file,str(e)))

############################################################################################
    
