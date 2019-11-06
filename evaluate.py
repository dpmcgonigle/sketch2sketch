#! /usr/bin/env python
"""
    evaluate.py

    This program will take two datasets and evaluate them based on the selected evaluation method

    See OPTIONS and FUNCTIONS sections for lists of each.

    Examples:   (d) means already a default
        ./evaluate.py --target_dir /mnt/d/data/sketch_data/testing/unaligned_sketchy_sm/val_targets --sketch_dir /mnt/d/data/sketch_data/checkpoints/fall/photosketch_5k1x/fake_A --debug all
"""
import argparse
import os, sys
import cv2
from glob import glob
import numpy as np
from datetime import datetime
import time
from skimage.util import random_noise
import inspect

############################################################################################
#                   O   P   T   I   O   N   S
#   --sketch_dir        ex: /mnt/d/data/sketch_data/checkpoints/fall/photosketch_5k1x/fake_A
#   --sketch_ext        png, jpg or gif
#   --target_dir        ex: /mnt/d/data/sketch_data/testing/unaligned_sket
#   --target_ext        png, jpg or gif
############################################################################################

############################################################################################
#                   F   U   N   C   T   I   O   N   S
#   List
#       str2bool
#       get_options
#       debug_functions
#       debug
#       num_images
#       load_images
#       get_filenames
#       load_image
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
    parser.add_argument('--debug', type=str, default=None, help="Comma-sep funcs (no space); e.g. func1,func2; 'all' for all functions")
    parser.add_argument('--sketch_ext', type=str, default="png", help="extension (jpg, png, gif) of the sketch")
    parser.add_argument('--target_ext', type=str, default="jpg", help="extension (jpg, png, gif) of the target")
    parser.add_argument('--sketch_dir', type=str, required=True, help="Required; generated image directory")
    parser.add_argument('--target_dir', type=str, required=True, help="Required; target image directory")
    parser.add_argument('--method', type=str, default='dice', help="Method of evaluation (dice, l1, l2)")
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
def debug(input_str, functions=debug_functions()):
    """ 
    Simple debugging print statements that can be turned off or on with specified functions. 
    inputs: 
        str, a string to output
        functions, a list of string function names you want to debug (this program uses debug_functions to get these from cmd line)
            ex: debug("Some output", functions=["get_source_dir","load_image"]), 
            or: debug("Other output", functions=debug_functions())
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
# END debug
############################################################################################

############################################################################################
def load_images(filepaths, opt):
    """
    returns list of numpy arrays representing the images stored at each path in filepaths.
    
    Params:
        filepaths(list of strings)      list of filepaths containing images to be returned
    """
    imgs = []
    for path in filepaths:
        assert os.path.exists(path), 'ERROR load_images(): filepath %s doesnt exist' % path

        binary = True if opt.method.lower() in ['dice'] else False
        imgs.append(load_image(path, binary=binary))

    return imgs
# END load_images
############################################################################################

############################################################################################
def dice(im1, im2, empty_score=1.0):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Borrowed from https://gist.github.com/brunodoamaral/e130b4e97aa4ebc468225b7ce39b3137
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        Both are empty (sum eq to zero) = empty_score
        
    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return (2. * intersection.sum() / im_sum)
# END dice
############################################################################################

############################################################################################
def get_filenames(imgdir, ext=None):
    """
    Returns a list of filenames (without path) from the data dir.  
    It is important that label images have same name (or at least carry the filename as a substring

    Params:
        ext(str)        image extension (png, jpg, gif, etc)
    """
    wild_card = "*"
    source_regex = os.path.join(imgdir, wild_card)
    debug("source_regex: %s" % source_regex)

    if ext:
        source_regex += ext
        
    # basename returns the filename without path
    filenames = [os.path.basename(f) for f in sorted(glob(source_regex))]
    debug("First filename with imgdir %s and ext %s: %s" % (imgdir, str(ext), filenames[0]))
    debug("Last filename with imgdir %s and ext %s: %s" % (imgdir, str(ext), filenames[-1]))

    return filenames
#   End get_filenames
############################################################################################

############################################################################################
def load_filename_sets(opt):
    """
    returns both lists of filenames (target and sketch) if all names in target dir are contained in sketch dir.
    """

    target_filenames = get_filenames(opt.target_dir, ext=opt.target_ext)
    sketch_filenames = get_filenames(opt.sketch_dir, ext=opt.sketch_ext)

    #   Ensure all images in target dir are in sketch_dir
    for f in target_filenames:
        sketch = os.path.splitext(f)[0] + '.%s'%opt.sketch_ext
        debug('TARGET: %s, SKETCH: %s'%(f,sketch))
        assert sketch in sketch_filenames, 'ERROR, sketch dir must contain all images in target dir.'

    debug('All TARGET images in %s found in %s' % (opt.target_dir, opt.sketch_dir))
    return target_filenames, sketch_filenames
# END load_filename_sets
############################################################################################

############################################################################################
def load_image(image_path, binary=True):
    """
    use cv2 module to load the grayscale of an image with binary values.
    output: numpy ndarray of image if it exists, with binary values.
    """

    if not os.path.isfile(image_path):
        raise OSError("Image %s not found" % image_path)

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if binary:
        ret, otsu = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
        #   Since we are interested in the "ink", which shows up as black, or 0,
        #   I am switching the ink to 1 and background to 0
        image=np.where(otsu > 0, 0, 1)

        dt = datetime.now().strftime('%H-%M-%S.%f')
        cv2.imwrite('./misc/img/%s.png'%dt, image)
    return image.astype(np.uint8)
# END load_image
############################################################################################

############################################################################################
def evaluate(target_images, sketch_images, method='dice'):
    """
    Performs evaluation metrics on two image sets representing sketches.

    Params:
        target_images (list of np arrays representing images)
        sketch_images (list of np arrays representing images)
        method (str)        dice, l1, l2, etc
    """
    evals = []
    for i in range(len(target_images)):
        func = getattr(sys.modules[__name__], method)
        debug('target_image shape: %s, sketch_image shape: %s' % (target_images[i].shape, sketch_images[i].shape))
        loss = func(target_images[i], sketch_images[i])
        evals.append(loss)

    return np.array(evals)
# END evaluate
############################################################################################

############################################################################################
#                               _   _   M   A   I   N   _   _                              #
############################################################################################
if __name__ == "__main__":

    #   Get command line options
    opt = get_options()
    debug('START PROGRAM\n======================')
    
    target_filenames, sketch_filenames = load_filename_sets(opt)

    target_images = load_images([os.path.join(opt.target_dir, f) for f in target_filenames], opt)
    sketch_images = load_images([os.path.join(opt.sketch_dir, f) for f in sketch_filenames], opt)

    #   Resize images, if necessary
    for i in range(len(target_images)):
        if target_images[i].shape != sketch_images[i].shape:
            sketch_images[i] = cv2.resize(sketch_images[i], target_images[i].shape)

    evals = evaluate(target_images, sketch_images, method=opt.method)
