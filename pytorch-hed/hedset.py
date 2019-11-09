#! /usr/bin/env python

###################################################################################
#       hedset.py
#   Perform HED on all images in a dataset
#   9/13/2019
#
#   EXAMPLE
#   ./hedset.py --datadir /data/data/sketch_data/sketchydb/sketchydb_2x --mode train --side B --procs 5
#   ./hedset.py --datadir /data/data/sketch_data/photo_sketching/photo_sketching_5k --mode train --side A --procs 2 --invert --binarize
###################################################################################

import os, sys
from subprocess import Popen, PIPE
from glob import glob
import multiprocessing as mp
from run import edge_aligned, edge_unaligned
import argparse

###################################################################################
#
#   Process Individual Image
#
###################################################################################
def process_image(source, dest, side, img_type, invert=False, binarize=False):
    print("======================================================================")
    if img_type == 'aligned':
        print("PID %d: EDGE MAP FOR %s ===> %s (side %s)" % (os.getpid(), source, dest, side))
    else:
        print("PID %d: EDGE MAP FOR %s ===> %s (%s)" % (os.getpid(), source, dest, img_type))
    #   arguments_strModel, arguments_strIn, arguments_strOut, arguments_strSide

    if img_type == "aligned":
        print('call: edge_aligned("bsds500", %s, %s, %s, %s)'%(source, dest, side, str(invert)))
        edge_aligned("bsds500", source, dest, side, arguments_invert=invert, arguments_binarize=binarize)
    elif img_type == "unaligned":
        print('call: edge_unaligned("bsds500", %s, %s, %s)'%(source, dest, str(invert)))
        edge_unaligned("bsds500", source, dest, arguments_invert=invert, arguments_binarize=binarize)
    """
    call = "python run.py --model bsds500 --in %s --out %s --side %s" % (image, os.path.join(newpath, imgname), side)
    print(call)
    p = Popen (call.split(), stdout = PIPE, stderr = PIPE)
    out, err = p.communicate()

    if out:
        print("STDOUT: %s" % out)
    if err:
        print("STDERR: %s" % err)
    """

###################################################################################
#
#       M       A       I       N
#
###################################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, required=True, default=None, help='Image dir (reqd)')
    parser.add_argument('--mode', type=str, default='train', help='[train|val|test]')
    parser.add_argument('--side', type=str, default='A', help='[A|B]')
    parser.add_argument('--img_type', type=str, default='aligned', help='[aligned|unaligned]')
    parser.add_argument('--procs', type=int, default=2, help='Number of processes (should be limited by amount of RAM on GPU; ~2GB per image I think.')
    parser.add_argument('--invert', action='store_true', help='Invert the black and white so that edges are black.')
    parser.add_argument('--binarize', action='store_true', help='Binarize the output by rounding.')
    args = parser.parse_args()

    #   Get CPUs - can't use cpu_count since I will run out of memory 
    #processes = mp.cpu_count() - 1 or 1
    processes = args.procs
    mode = args.mode
    side = args.side
    dirname = args.datadir
    img_type = args.img_type
    invert = args.invert
    binarize = args.binarize

    #   Check Args
    assert os.path.exists(dirname), "Directory %s doesn't exist" % dirname
    assert mode in ["train", "val", "test"], "Mode %s not in ['train', 'val', 'test']" % mode
    assert side in ["A", "B"], "side %s not in ['A', 'B']" % side

    #   Get images
    oldpath = os.path.join(dirname, mode)
    images = glob(os.path.join(oldpath, "*"))

    #   Create destination directory
    newpath = os.path.join("%s_hed" % dirname, mode)
    print("Creating directory %s" % (newpath))
    try:
        os.makedirs(newpath)
    except:
        print("%s already exists!" % newpath)

    #   Loop through images to instantiate processes
    pool = mp.Pool(processes=processes)

    for count,image in enumerate(images):

        source = image
        dest = os.path.join(newpath, os.path.basename(image))

        process_image(source, dest, side, img_type, invert)

        #process_image(source, dest, side)
        #proc = mp.Process(target=process_image, args=(source, dest, side, ))
        pool.apply_async(process_image, args=(source, dest, side, img_type, invert, binarize, ))
        #procs.append(proc)
        #proc.start()
        #if count > 10:
        #    break

    pool.close()
    pool.join()
    """
    #   Complete the processes
    for proc in procs:
        proc.join()
    """
