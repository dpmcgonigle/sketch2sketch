#! /usr/bin/env python
"""
    hybrid_hed.py

        This program creates a hybrid of the hed image and original image for a dataset

    Examples:   (d) means already a default
        ./hybrid_hed.py --dataroot /data/data/sketch_data/photo_sketching --dataset ps5k_aug --phase train --scale_img 1.0 --suffix fullhybrid --side A --debug all
        ./hybrid_hed.py --dataroot /data/data/sketch_data/testdata --dataset aligned_sketchy_sm --phase val --scale_img 0.5 --suffix halfhybrid --side B --debug all
"""
import argparse
import os, sys
import inspect
from glob import glob
import numpy as np
import cv2

############################################################################################
#                   F   U   N   C   T   I   O   N   S
#   List
#       str2bool
#       get_options
#       debug_functions
#       print_debug
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
    parser.add_argument('--dataset', type=str, default="photo_sketching/ps5k_aug", help="Should have <opt.phase> dir and parallel hed dir")
    parser.add_argument('--phase', type=str, default="train", help="train, test, val, etc - should be a directory in data_dir")
    parser.add_argument('--dataroot', type=str, default="/data/data/sketch_data") # expects directory 'training'
    parser.add_argument('--suffix', type=str, default="hybrid", help="suffix for new directory")
    parser.add_argument('--scale_img', type=float, default=0.5, help="how much to scale the original image")
    parser.add_argument('--imgwidth', type=int, default=256, help="pixel width of image")
    parser.add_argument('--side', type=str, default="A", help="input image side")
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
#                               _   _   M   A   I   N   _   _                              #
############################################################################################
if __name__ == "__main__":

    #   Get command line options
    opt = get_options()
    
    path = os.path.join(opt.dataroot, opt.dataset, opt.phase)
    hedpath = os.path.join(opt.dataroot, opt.dataset + '_hed', opt.phase)
    newpath = os.path.join(opt.dataroot, opt.dataset + '_%s'%opt.suffix, opt.phase)

    fnames = sorted(glob(path + '/*'))
    hednames = sorted(glob(hedpath + '/*'))

    try:
        os.makedirs(newpath)
    except:
        pass

    for i in range(len(fnames)):
        #print('fname: %s, hedname: %s' % (fnames[i], hednames[i]))
        img = cv2.imread(fnames[i], cv2.IMREAD_GRAYSCALE).astype(np.uint8)

        if opt.side == 'A':
            img = np.hstack([np.invert(img[:,:opt.width]), img[:,opt.width:]])
        elif opt.side == 'B':
            img = np.hstack([img[:,:256], np.invert(img[:,256:])])
        else:
            print('ERROR: --side needs to be A or B')
            exit(1)

        hedimg = cv2.imread(hednames[i], cv2.IMREAD_GRAYSCALE).astype(np.uint8)
        #hedimg = np.hstack([np.invert(hedimg[:,:256]), hedimg[:,256:]])
        avg = np.maximum(hedimg, (img*opt.scale_img))

        cv2.imwrite(os.path.join(newpath, os.path.basename(hednames[i])), avg)
