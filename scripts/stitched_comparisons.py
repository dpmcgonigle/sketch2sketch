#! /usr/bin/env python
"""
    stitched_comparisons.py

    This program will take the stitched images from each directory and mash them into a grid for comparisons.  Original format:
    Image will be in a 3 row by 2 column grid, each picture 256 x 256:
        [   Orig A,         Orig B      ] 
        [   Processed A,    Processed B ]
        [   Generated A,    Generated B ]

    Examples:   (d) means already a default
        ./stitched_comparisons.py --exclude summer,*fruit*,*insect*,google --num_imgs 10 --rows 2 --subdir val_stitched --debug all

"""
from datetime import datetime
import argparse
import os, sys
import inspect
from glob import glob
import cv2
import numpy as np
############################################################################################
#                   O   P   T   I   O   N   S
#   --exclude
#   --num_imgs
#   --rows
#   --basedir
#   --subdir
#   --ext
#   --debug
#   --outputdir
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
    parser.add_argument('--basedir', type=str, default=".", help="Base directory to root around in for image directories")
    parser.add_argument('--subdir', type=str, default="val_stitched", help="Sub-directory of each directory to look for images in")
    parser.add_argument('--outputdir', type=str, default="./stitch_jobs", help="Directory to put all the output images in")
    parser.add_argument('--exclude', type=str, default="summer", help="Comma-delimited list of directories to exclude from this process")
    parser.add_argument('--num_imgs', type=int, default=10, help="Number of rows of images")
    parser.add_argument('--rows', type=int, default=2, help="Total number of images to cobble together")
    parser.add_argument('--ext', type=str, default="png", help="Image extension to look for")
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

    #   Extract exclusion list
    exclude_list = opt.exclude.split(',')
    flatten = lambda l: [item for sublist in l for item in sublist]
    exclude_glob = flatten([glob(g) for g in exclude_list])
    print_debug("\n===========Excluding directories:\n%s" % "\n".join(exclude_glob))

    #   Get list of directories to trudge through
    dirs = [d for d in os.listdir(opt.basedir) if os.path.isdir(d) and d not in exclude_glob]
    print_debug("\n===========Using directories:\n%s" % "\n".join(dirs))

    #   Make outputdir if it doesn't exist
    if not os.path.isdir(opt.outputdir):
        os.mkdir(opt.outputdir)

    #   Trudge through directories
    for d in dirs:
        workdir = os.path.join(d,opt.subdir)
        print_debug("Entering workdir %s" % workdir)
        if not os.path.isdir(workdir):
            print("ERROR: Directory %s not found.  Continuing to next directory" % workdir)
            continue

        imgs = glob(os.path.join(workdir,"*%s"%opt.ext))
        if len(imgs) < opt.num_imgs:
            print("ERROR: Number of images in %s less than %d. Continuing to next directory" % (workdir, opt.num_imgs))
            continue

        workimgs = imgs[30:30+opt.num_imgs]
        #workimgs = imgs[:opt.num_imgs]
        print_debug("\n===========Using images:\n%s" % "\n".join(workimgs))

        imgs_per_row = int(opt.num_imgs/opt.rows)
        x = [workimgs[i*imgs_per_row:(i+1)*imgs_per_row] for i in range(opt.rows)] 
        try:
            stitched_img = np.vstack(
                [np.hstack([cv2.imread(img) for img in workimgs[i*imgs_per_row:(i+1)*imgs_per_row]]) for i in range(opt.rows)]
            )

            #   Add header space for text (has to have three color channels
            #print("stitched_img shape %s" % str(stitched_img.shape))
            header = np.zeros((80,stitched_img.shape[1],3))
            #print("header shape %s" % str(header.shape))
            final_img = np.vstack([header, stitched_img])

            #   Add text to stitched image (image, text, (x, y), font, color, style)
            cv2.putText(final_img, d, (60, 60), cv2.FONT_HERSHEY_DUPLEX, 3.0, (255, 255, 255), lineType=cv2.LINE_AA)

            #   Write image
            cv2.imwrite(os.path.join(opt.outputdir,"%s.%s"%(d,opt.ext)), final_img)

        except Exception as e:
            print("ERROR (Exit with code 1): While trying to stitch images - %s" % str(e))
            exit(1)
