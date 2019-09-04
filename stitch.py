# Crops all images in a directory from 512 x 256 x 3 to 256 x 256 x 3

import cv2, os, sys
import numpy as np

project = sys.argv[1]

target = "D:\\Data\\sketch_data\\testing\\val_targets"
real = "D:\\Data\\sketch_data\\testing\\val"
val_A = "D:\\Data\\sketch_data\\checkpoints\\%s\\val_A" % project
val_B = "D:\\Data\\sketch_data\\checkpoints\\%s\\val_B" % project
stitch = "D:\\Data\\sketch_data\\checkpoints\\%s\\stitch" % project

if any ([not os.path.isdir(x) for x in [target, real, val_A, val_B]]):
    print("Not all paths worked.  Exiting")
    exit(1)

if not os.path.isdir(stitch):
    os.makedirs(stitch)

files = os.listdir(target)

for f in files:
    try:
        print(f)
        fpng = f[:-3] + "png"
        #print(fpng)

        # real   cv2.resize(img,(128,128))
        realimg = cv2.resize( cv2.imread(os.path.join(real, f)) ,(128,128))
        #print("realimg shape: %s" % str(realimg.shape))
        
        # canny
        cannyimg = cv2.imread(os.path.join(val_A, "real_A_%s" % fpng))
        #print("cannyimg shape: %s" % str(cannyimg.shape))
        
        # prediction GAN A
        pred_A = cv2.imread(os.path.join(val_B, "fake_B_%s" % fpng))
        #print("predimg shape: %s" % str(predimg.shape))
        
        # prediction GAN B
        pred_B = cv2.imread(os.path.join(val_A, "fake_B_%s" % fpng))
        #print("predimg shape: %s" % str(predimg.shape))
        
        # target
        targetimg = cv2.resize( cv2.imread(os.path.join(target, f)) ,(128,128))
        #print("targetimg shape: %s" % str(targetimg.shape))
        
        stitched_img = np.hstack([realimg, cannyimg, pred_A, pred_B, targetimg])
        #print("stitched_img shape: %s" % str(stitched_img.shape))
        
        cv2.imwrite(os.path.join(stitch,f), stitched_img)
    except Exception as e:
        print("Error with %s: %s" % (f, str(e)))
    