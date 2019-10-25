#! /usr/bin/env python

###################################################################################
#       hedset.py
#   Perform HED on all images in a dataset
#   9/13/2019
#
#   EXAMPLE
#   ./hedset.py /data/data/sketch_data/sketchydb/sketchydb_2x train B
#   ./hedset.py /data/data/sketch_data/photo_sketching/photo_sketching_5k train A
###################################################################################

import os, sys
from subprocess import Popen, PIPE
from glob import glob
import multiprocessing as mp
from run import edge

###################################################################################
#
#   Process Individual Image
#
###################################################################################
def process_image(source, dest, side):
    print("======================================================================")
    print("PID %d: EDGE MAP FOR %s ===> %s (side %s)" % (os.getpid(), source, dest, side))
    print("call = edge(\"%s\", \"%s\", \"%s\", \"%s\")" % ("bsds500", source, dest, side))
    #   arguments_strModel, arguments_strIn, arguments_strOut, arguments_strSide
    edge("bsds500", source, dest, side)
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

    #   Get CPUs
    #processes = mp.cpu_count() - 1 or 1
    processes = 5

    #   Get Args
    try:    
        dirname = sys.argv[1]
        mode = sys.argv[2]
        side = sys.argv[3]
    except Exception as e:
        print("ERROR taking in dataset dir (arg 1), mode (arg 2; train, val, test) and side (arg 3; A or B): %s.  Exiting with code 1." % str(e))
        exit(1)

    #   Check Args
    assert os.path.exists(dirname), "Directory %s doesn't exist" % dirname
    assert mode in ["train", "val", "test"], "Mode %s not in ['train', 'val', 'test']" % mode
    assert side in ["A", "B"], "side %s not in ['A', 'B']" % side

    #   Get images
    oldpath = os.path.join(dirname, mode)
    images = glob(os.path.join(oldpath, "*"))

    #   Create destination directory
    newpath = os.path.join("%s_hed" % (dirname, mode))
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
        #process_image(source, dest, side)
        #proc = mp.Process(target=process_image, args=(source, dest, side, ))
        pool.apply_async(process_image, args=(source, dest, side, ))
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
