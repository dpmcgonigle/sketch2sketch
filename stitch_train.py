# stitches together training images from repo (command line arg) in [realimg, real_A, fake_A, real_B, fake_B] format

import cv2, os, sys
import numpy as np
from glob import glob
import logging

# Logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

project = sys.argv[1]
dataset = sys.argv[2]
size = int(sys.argv[3])

real = "D:\\Data\\sketch_data\\%s\\train" % dataset
train = "D:\\Data\\sketch_data\\checkpoints\\%s\\train" % project
stitch = "D:\\Data\\sketch_data\\checkpoints\\%s\\stitch_train" % project

if any ([not os.path.isdir(x) for x in [real, train]]):
    print("Not all paths worked.  Exiting")
    exit(1)

if not os.path.isdir(stitch):
    os.makedirs(stitch)

files = glob("%s\\*fake_A*" % train)

for f in files:
    try:
        f = os.path.basename(f)
        # format: epoch_100_iter_01171000_fake_A_24-6.png
        elems = f.split('_')
        start = "_".join(elems[:4])
        end = elems[-1]
        img_name = end.split('.')[0]
        print("img_name: %s" % img_name)
        
        #print("start: %s" % start)
        #print("end: %s" % end)
        #print("img_name: %s" % img_name)
        
        print(f)
        
        # real   cv2.resize(img,(128,128))
        realimgName = os.path.join(real, "%s.jpg" % img_name)
        print("realimgName: %s" % realimgName)
        realimg = cv2.resize( cv2.imread(realimgName)[:,256:,:] ,(size,size))
        print("realimg shape: %s" % str(realimg.shape))
        
        # real_A
        real_A = cv2.imread(os.path.join(train, "%s_real_A_%s" % (start,end)))
        print("real_A shape: %s" % str(real_A.shape))
        
        # fake_A
        fake_A = cv2.imread(os.path.join(train, f))
        print("fake_A shape: %s" % str(fake_A.shape))
        
        # real_B
        real_B = cv2.imread(os.path.join(train, "%s_real_B_%s" % (start,end)))
        print("real_B shape: %s" % str(real_B.shape))
        
        # fake_B
        fake_B = cv2.imread(os.path.join(train, "%s_fake_B_%s" % (start,end)))
        print("fake_B shape: %s" % str(fake_B.shape))
        
        stitched_img = np.hstack([realimg, real_A, fake_A, real_B, fake_B])
        print("stitched_img shape: %s" % str(stitched_img.shape))
        
        cv2.imwrite(os.path.join(stitch,f), stitched_img)
    except Exception as e:
        print(logger.exception(e))
    