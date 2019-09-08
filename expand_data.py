#! /usr/bin/env python
"""
    expand_data.py

    This program will take a dataset and:
        (1) augment the images with:
            - Horizontal and Vertical flips (one or the other)
            - Rotations
            - Translations 
            - Top and bottom hat transforms (one or the other)
            - Random Gaussian and Salt & Pepper Noise (one or the other)
        (2) expand the dataset by a multiplier (e.g. 1,000 image dataset becomes 2,000 with #2)

    See OPTIONS and FUNCTIONS sections for lists of each.

    Examples:   (d) means already a default
        (expand a dataset by 5X, performing each augmentation at a rate of 40%; 40% translate, 40% noise, 40% flip, etc)
        ./expand_dataset.py --dataset <dataset> --dataset_multiplier 5 --augmentation_threshold 0.4 --image_type grayscale
        (expand a dataset by 2X, only on png images and only performing rotations at a rate of 100%
        ./expand_dataset.py --dataset <dataset> --augmentation_threshold 1.0 --ext png --translate false --tophat false --noise false --flip false
        ./expand_data.py --dataset photo_sketching_5k --augmentation_threshold 0.8 --dataset_multiplier 4
        Zhou's account on Seahawks
        python3 expand_data.py --dataset photo_sketching/photo_sketching_5k --augmentation_threshold 0.8 --dataset_multiplier 2 --debug all --data_dir /ihome/zhe/McGonigle/data/sketch_data

"""
import argparse
import os, sys
import cv2
from glob import glob
import numpy as np
from datetime import datetime
from skimage.util import random_noise
import inspect

############################################################################################
#                   O   P   T   I   O   N   S
#   '--aligned'        '--dataset'      '--phase'       '--ext'
#   '--dataset_multiplier'              '--data_dir'    '--label_dir'
#   '--augmentation_threshold'          '--image_type'  '--flip'
#   '--rotate'         '--translate'    '--tophat'      '--noise'
#   '--debug'
#       
############################################################################################

############################################################################################
#                   F   U   N   C   T   I   O   N   S
#   List
#       str2bool
#       get_options
#       debug_functions
#       print_debug
#       get_source_dir
#       get_dest_dir
#       num_images
#       load_images
#       get_filenames
#       load_image
#       load_image_batch
#       augment_imageset
#       expand_dataset
#       
############################################################################################

############################################################################################
def str2bool(v):
    """
    converts command line input to boolean representation ('yes', 'true', 't', 'y', '1')
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
# end str2bool
############################################################################################

############################################################################################
def get_options():
    """
    return command line options as an argparse.parse_args() object
    if called from jupyter, 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', type=str, default=None, help="Comma-sep func list (no spaces); e.g. func1,func2; 'all' for all functions")
    parser.add_argument('--aligned', type=str2bool, default=True, help="If pictures are attached, select this option")
    parser.add_argument('--dataset', type=str, default="photo_sketching", help="Dataset - should have a <opt.phase> directory in it")
    parser.add_argument('--phase', type=str, default="train", help="train, test, val, etc - should be a directory in data_dir")
    parser.add_argument('--ext', type=str, help="extension (jpg, png, gif, etc -- optional)")
    parser.add_argument('--img_width', type=int, default=256, help="used to split aligned images")
    parser.add_argument('--dataset_multiplier', default=2, type=int, help="how many total images per original in the resulting dataset")
    parser.add_argument('--data_dir', type=str, default="/data/data/sketch_data") # expects directory 'training'
    parser.add_argument('--label_dir', type=str, help="If the x and y pictures aren't aligned (attached), you need to supply the label_dir")
    parser.add_argument('--augmentation_threshold', type=float, default=0.5, help="randomly perform one of the augmentation procedures this % of the time")
    parser.add_argument('--image_type', type=str, default="rgb", help="rgb or grayscale")
    parser.add_argument('--batch_size', type=int, default=250, help="Size of image batches; helpful for preventing MemoryErrors")
    parser.add_argument('--flip', type=str2bool, default=True)
    parser.add_argument('--rotate', type=str2bool, default=True)
    parser.add_argument('--translate', type=str2bool, default=True)
    parser.add_argument('--tophat', type=str2bool, default=False)
    parser.add_argument('--noise', type=str2bool, default=True)
    # If you want to call this get_args() from a Jupyter Notebook, you need to uncomment -f line. Them's the rules.
    #parser.add_argument('-f', '--file', help='Path for input file.')
    return parser.parse_args()
############################################################################################
    
############################################################################################
def debug_functions():
    """
    Parse the command-line arguments to get the list of functions to debug.
    When running this script, pass in a comma-separated list (NO SPACES) of functions you'd like to debug.
    Use "main" for main program body execution debugging.
        ex: ./expand_data.py <some options> --debug load_images,augment_imageset,main
    """
    #   Get command line options
    opt = get_options()
    if opt.debug:
        function_list = opt.debug.split(',')
        return function_list
    else:
        return None

############################################################################################

############################################################################################
def print_debug(input_str, functions=debug_functions()):
    """ 
    Simple debugging print statements that can be turned off or on with specified functions. 
    inputs: 
        str, a string to output
        functions, a list of string function names you want to debug (this program uses debug_functions to get these from cmd line)
            ex: print_debug("Some output", functions=["get_source_dir","load_image"]), 
            or: print_debug("Other output", functions=debug_functions())
    """
    #   if the --debug option isn't used, don't do anything!
    if not opt.debug:
        return

    caller_function = inspect.stack()[1].function
    if caller_function == "<module>":
        caller_function = "main"
    if "all" in functions or caller_function in functions: 
        time_stamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        print("%s %s(): %s" % (time_stamp, caller_function, input_str))
# END print_debug
############################################################################################

############################################################################################
def get_source_dir(opt):
    """ Return default data_dir if one isn't provided - should point to root data folder """
    return os.path.join(opt.data_dir, opt.dataset, opt.phase)
#   End get_source_dir
############################################################################################

############################################################################################
def get_dest_dir(opt):
    """ Return default data_dir if one isn't provided - should point to root data folder """
    return os.path.join(opt.data_dir, opt.dataset + "_%dx" % opt.dataset_multiplier, opt.phase)
#   End get_source_dir
############################################################################################

############################################################################################
def num_images(opt):
    """
    returns number of images in the source directory
    """
    wild_card = "*"
    if opt.ext:
        source_regex += opt.ext
        
    source_regex = os.path.join(get_source_dir(opt), wild_card)
    print_debug("source_regex: %s" % source_regex)
        
    # basename returns the filename without path
    filenames = [f for f in sorted(glob(source_regex))]

    return len(filenames)
############################################################################################

############################################################################################
def load_images(opt, batch=0):
    """
    returns tuple of data and filename dictionaries:
        data_images: dict_keys(['y_images', 'test_x_images'])
            numpy ndarrays ready for training and validation (dims N x C x H x W)
        data_names: dict_keys(['filenames']) - lists of basenames without extensions
    image_type is optional and can be str ('grayscale', 'rgb'); default grayscale
    """
    # Get filenames for X and Y train and val images
    x_data_paths, y_data_paths = load_data_filenames(opt, batch=batch)
    print_debug("First x_data_path: %s" % x_data_paths[0])
    print_debug("Last x_data_path: %s" % x_data_paths[-1])
    if(y_data_paths):
        print_debug("First y_data_path: %s" % y_data_paths[0])
        print_debug("Last y_data_path: %s" % y_data_paths[-1])
        
    # Load X and Y train and val images as np arrays
    # If grayscale, will come in (n x h x w) format, so it will need to be expanded to (n x c x h x w)
    x_images, y_images = load_image_batch(opt, x_data_paths, y_data_paths)
        
    # ensure all images were of the same size; if not, load_image_batch will return a 1d array of type object
    #assert (x_images.ndim > 1 and y_images.ndim > 1 and test_x_images.ndim > 1 and test_y_images.ndim > 1),\
    #    "Ensure images are all same size; look in dataset with 1 dimension: x=>%d, y=>%d, \
    #    test_x=>%d, test_y=>%d" % (x_images.ndim, y_images.ndim, test_x_images.ndim, test_y_images.ndim)

    data_images = {"x_images": x_images,
                   "y_images": y_images}
    
    data_names = {"x_data_paths": x_data_paths,
                    "y_data_paths": y_data_paths}

    return data_images, data_names
# END load_test_images
############################################################################################

############################################################################################
def get_filenames(imgdir, batch=0, ext=None):
    """
    Returns a list of filenames (without path) from the data dir.  
    It is important that label images have same name (or at least carry the filename as a substring
    """
    wild_card = "*"
    if ext:
        source_regex += ext
        
    source_regex = os.path.join(imgdir, wild_card)
    print_debug("source_regex: %s" % source_regex)
        
    # basename returns the filename without path
    filenames = [f for f in sorted(glob(source_regex))]
    print_debug("First filename: %s" % filenames[0])
    print_debug("Last filename: %s" % filenames[-1])

    #   get the indexes for the start and end of this image batch    
    start = batch*opt.batch_size
    end = (batch+1)*opt.batch_size-1
    if end >= num_images(opt):
        end = num_images(opt) - 1

    return filenames[ start : end ]
#   End get_filenames
############################################################################################

############################################################################################
def load_data_filenames(opt, batch=0):
    """
    returns 1 or 2 lists of filenames: x_image_paths, y_image_paths(if images aren't aligned)
    The lists will be in a random permutation of the number of images; uses constant random seed for reproducibility.
    """

    data_image_files = get_filenames(get_source_dir(opt), batch=batch, ext=opt.ext)
    print_debug("First data_image_file: %s" % data_image_files[0])
    print_debug("Last data_image_file: %s" % data_image_files[-1])
    if opt.label_dir:
        data_label_files = get_filenames(opt.label_dir, batch=batch, ext=opt.ext)
        assert len(data_image_files) == len(data_label_files), "X and Y datasets have different number of images"
    else:
        data_label_files = None
    
    return data_image_files, data_label_files
# END load_data_filenames
############################################################################################

############################################################################################
def load_image(image_path, image_type="grayscale"):
    """
    use cv2 module to load an image
    inputs: image_path, image_type (optional, str: "grayscale" or "rgb")
    output: numpy ndarray of image if it exists
    """

    if not os.path.isfile(image_path):
        raise OSError("Image %s not found" % image_path)
    if image_type.lower() == "grayscale":
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    elif image_type.lower() == "rgb":
        image = cv2.imread(image_path, -1)
    else:
        raise ValueError("image_type must be \"grayscale\", \"rgb\"...")
    return image.astype(np.uint8)
# END load_image
############################################################################################

############################################################################################
def load_image_batch(opt, x_data_paths, y_data_paths):
    """
    use load_image (which uses cv2 module to return a numpy ndarray) to load a numpy array of images.
    inputs: image_paths: list of filenames, image_type (optional, "grayscale" or "rgb")
    load_image loads numpy ndarray of images of shape (Height x Width x Channels[if more than 1])
    returns tuple of (x_images, y_images) numpy arrays
    """
    print_debug("First x_data_path: %s" % x_data_paths[0])
    print_debug("Last x_data_path: %s" % x_data_paths[-1])
    if(y_data_paths):
        print_debug("First y_data_path: %s" % y_data_paths[0])
        print_debug("Last y_data_path: %s" % y_data_paths[-1])
    x_images, y_images = [], []
    
    # load aligned images
    if opt.aligned:
        for path in x_data_paths:
            img_set = load_image(image_path=path, image_type=opt.image_type)
            x_images.append(img_set[:, :opt.img_width])
            y_images.append(img_set[:, -opt.img_width:])
    
    # Not aligned
    else:     
        for path in x_data_paths:
            x_images.append(load_image(image_path=path, image_type=opt.image_type))
        for path in y_data_paths:
            y_images.append(load_image(image_path=path, image_type=opt.image_type))
            
    return np.asarray(x_images), np.asarray(y_images)
# END load_image_batch
############################################################################################

############################################################################################
def augment_imageset(input_img, label_img, probability_threshold=0.2, flip=True, rotate=True, translate=True, tophat=True, noise=True):
    """
    Perform morphological transformations and other data augmentation methods on an input image and label image.
    Decide whether to perform each operation randomly based on the probability_threshold
    Operations:
    - Horizontal and Vertical flips (one or the other)
    - Rotations
    - Translations 
    - Top and bottom hat transforms (one or the other)
    - Random Gaussian and Salt & Pepper Noise (one or the other)
    Input:
    - input_img - of shape (height, width, num_channels); ***data should be in integer 0-255 format***
    - label_img - of shape (height, width, num_channels); ***data should be in integer 0-255 format***
    - probability_threshold - 0.0 to 1.0 (0% to 100%)
    """
    # Randomize
    np.random.seed()
    
    # Copy Arrays so that we don't run into any array shared storage issues
    x_img = input_img.copy()
    y_img = label_img.copy()
    rows,cols = x_img.shape[0] , x_img.shape[1] 
    channels = x_img.shape[-1]
    
    # Flips
    if flip and np.random.rand() <= probability_threshold:
        # Vertical flip is axis 0, Horizontal flip is axis 1
        flip_axis = 0 if np.random.rand() <= 0.5 else 1
        print_debug("flip axis %d"%flip_axis)
        for i in range(channels):
            x_img[:,:,i] = cv2.flip(x_img[:,:,i], flip_axis)
            y_img[:,:,i] = cv2.flip(y_img[:,:,i], flip_axis)
    
    # Rotations - important to put this before translate
    if rotate and np.random.rand() <= probability_threshold:
        # Rotate from -360 to +360 degrees
        rotation = np.random.randint(-360,360)
        print_debug("rotate %d degrees" % rotation)

        for i in range(channels):
            #(col/2,rows/2) is the center of rotation for the image 
            # M is transformation matrix (computer graphics concept)
            M = cv2.getRotationMatrix2D((cols/2,rows/2),rotation,1) 
            x_img[:,:,i] = cv2.warpAffine(x_img[:,:,i],M,(cols,rows), borderValue=(255,255,255))   
            y_img[:,:,i] = cv2.warpAffine(y_img[:,:,i],M,(cols,rows), borderValue=(255,255,255))  
        
    # Translations
    if translate and np.random.rand() <= probability_threshold:
        # Translate from -(1/4) to +(1/4) of image height/width
        translation_x = np.random.randint(int(- x_img.shape[1] / 4), int(x_img.shape[1] / 4))
        translation_y = np.random.randint(int(- x_img.shape[0] / 4), int(x_img.shape[0] / 4))
        print_debug("translate [%d, %d] pixels" % (translation_x, translation_y))
        
        for i in range(channels):
            # M is transformation matrix (computer graphics concept)
            M = np.float32([[1,0,translation_x],[0,1,translation_y]]) 
            x_img[:,:,i] = cv2.warpAffine(x_img[:,:,i],M,(cols,rows), borderValue=(255,255,255))         
            y_img[:,:,i] = cv2.warpAffine(y_img[:,:,i],M,(cols,rows), borderValue=(255,255,255))  
        
    # Tophat transforms
    if tophat and np.random.rand() <= probability_threshold:
        # tophat (image opening) is black_tophat; bottomhat (image closing) is white_tophat
        # https://www.youtube.com/watch?v=P2vAhqGgV44
        # Size 25 square for structuring element visually looks like a good choice for this type of data
        xform = morphology.black_tophat if np.random.rand() <= 0.5 else morphology.white_tophat
        print_debug("Tophat transform: %s" % str(xform))
        for i in range(channels):
            x_img[:,:,i] = xform(x_img[:,:,i], selem=morphology.square(25))

    # Make some noise
    if noise and np.random.rand() <= probability_threshold:
        # random_noise() uses scale [0, 1.0], will need to multiply to get it to [0, 255]
        # inherently it use np.random.normal() to create normal distribution and adds the generated noised back to image
        # modes are 'gaussian', 'localvar', 'poisson', 'salt', 'pepper', 's&p', and 'speckle'
        # var stands for variance of the distribution. Used in 'gaussian' and 'speckle'. amount can be used for s&p
        # variance / amount will be up to 0.25
        noise_amount = np.random.randint(0,25) / 100.0
        mode = "s&p" if np.random.rand() <= 0.5 else "gaussian"
        print_debug("Adding %.02f %s noise" % (noise_amount, mode))
        
        noise_img = None
        for i in range(channels):
            if mode == "gaussian":
                noise_img = random_noise(x_img[:,:,i], mode=mode, var=noise_amount**2)  
            elif mode == "s&p":
                noise_img = random_noise(x_img[:,:,i], mode=mode, amount=noise_amount)
            x_img[:,:,i] = (255*noise_img).astype(np.uint8)
        
    return x_img, y_img
# END augment_imageset
############################################################################################

############################################################################################
def expand_dataset(opt, x_images, y_images):
    """
    loop through dataset and create augmented images to return as the expanded dataset
    x_images and y_images are numpy arrays containing the image pixel data
    """
    print_debug("x_images type: %s, shape: %s" % (str(type(x_images)), str(x_images.shape)))
    print_debug("y_images type: %s, shape: %s" % (str(type(y_images)), str(y_images.shape)))
    new_x_set, new_y_set = [], []
    
    for i in range(len(x_images)):
        for j in range(opt.dataset_multiplier):
            x_img, y_img = augment_imageset(x_images[i], y_images[i], probability_threshold=opt.augmentation_threshold, 
                flip=opt.flip, rotate=opt.rotate, translate=opt.translate, tophat=opt.tophat, noise=opt.noise)
                
            new_x_set.append(x_img)
            new_y_set.append(y_img)
                
    return np.asarray(new_x_set), np.asarray(new_y_set)
#   end expand_dataset
############################################################################################

############################################################################################
def save_expanded_set(opt, new_x_images, new_y_images, x_data_paths, y_data_paths):
    """
    Save the expanded dataset!
    """
    destdir = get_dest_dir(opt)
    print_debug("destdir %s" % destdir)
    
    if not os.path.isdir(destdir):
        os.makedirs(destdir)
    
    if opt.aligned:
        for i in range(len(x_data_paths)):
            #   Get basename from path
            fname, ext = os.path.splitext( os.path.basename(x_data_paths[i]) )
        
            for j in range(opt.dataset_multiplier):
            
                #   index of the new image set is calculated based on the dataset multiplier and old image set length
                index = (i * opt.dataset_multiplier) + j
                
                #   stitch images together
                stitched_img = np.hstack([new_x_images[index], new_y_images[index]])
                
                #   save images
                cv2.imwrite(os.path.join(destdir, fname + "-%d%s" % (j,ext)), stitched_img)
                
#   end save_expanded_dataset
############################################################################################
#                               _   _   M   A   I   N   _   _                              #
############################################################################################
if __name__ == "__main__":

    #   Get command line options
    opt = get_options()
    
    #
    #   LOOP THROUGH IMAGE BATCHES OF SIZE --batch_size
    #
    for batch in range(int(np.ceil(num_images(opt) / opt.batch_size))):
        #   Get hash of images  {"x_images":np.array(n , h , w , c), "y_images":np.array(n , h , w , c)}
        #   and filenames       {"x_data_paths": [list of paths], "y_data_paths": [another list of paths]}
        data_images, data_names = load_images(opt, batch=batch)

        #   (Gotta keep em separated)
        x_images, y_images = data_images["x_images"], data_images["y_images"]
        x_data_paths, y_data_paths = data_names["x_data_paths"], data_names["y_data_paths"]
    
        #   Expand dataset
        new_x_images, new_y_images = expand_dataset(opt, x_images, y_images)
    
        #   Save expanded dataset
        save_expanded_set(opt, new_x_images, new_y_images, x_data_paths, y_data_paths)
