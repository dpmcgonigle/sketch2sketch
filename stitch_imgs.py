#! /usr/bin/env python
"""
    expand_data.py

    This program will take command-line arguments regarding images from training or testing and stitch them together:
    Image will be in a 3 row by 2 column grid, each picture 256 x 256:
        [   Orig A,         Orig B      ] 
        [   Processed A,    Processed B ]
        [   Generated A,    Generated B ]

    Examples:   (d) means already a default
        ./stitch_imgs.py --dataset photo_sketching_5k --name photosketch_5k1x --phase train --debug all
        ./stitch_imgs.py --dataset photo_sketching/photo_sketching_5k_8x --name photosketch_5k8x --phase train --debug all

"""
import argparse
import os, sys
import inspect
from util import util

############################################################################################
#                   O   P   T   I   O   N   S
#   --dataset
#   --dataset_mode
#   --name
#   --phase
#   --dataroot
#   --checkpoints_dir
#   --debug
#       
############################################################################################

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
    parser.add_argument('--dataset', type=str, default="photo_sketching", help="Dataset - should have a <opt.phase> directory in it")
    parser.add_argument('--dataset_mode', type=str, default="aligned", help="Dataset mode - [unaligned | aligned | single | colorization]")
    parser.add_argument('--phase', type=str, default="train", help="train, test, val, etc - should be a directory in data_dir")
    parser.add_argument('--dataroot', type=str, default="/data/data/sketch_data") # expects directory 'training'
    parser.add_argument('--checkpoints_dir', type=str, default="/data/data/sketch_data/checkpoints", help='models are saved here')
    parser.add_argument('--name', type=str, help='name of the experiment. It decides where to store samples and models')
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
    
    if opt.phase == "train":
        util.stitch_training_imgs(opt)
