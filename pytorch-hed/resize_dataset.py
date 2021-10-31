#! /usr/bin/env python
# Stitch original images to sketch images

from glob import glob
import cv2, os, sys
import numpy as np
import traceback
import math
import argparse

##############################################
def resize_img(img, height=256, width=256):
    """
    given input img and sketch, output an aligned imageset of [img|sketch]
    img and sketch should be of shape (c x h x w), as fetched from cv2.imread()
    """
    # get hw dims of img
    img_h, img_w, img_c = img.shape[0], img.shape[1], img.shape[2]
    
    # get axis height and width scales in relation to target height and width
    img_scaler_h = height / img_h
    img_scaler_w = width / img_w
    
    # scale image based on smallest scaling ratio (ensures no whitespace when we go to crop)
    img_scaler_axis = 0 if img_scaler_h > img_scaler_w else 1 # scaler axis is axis that 
    img_scaler = img_scaler_h if img_scaler_axis == 0 else img_scaler_w
    new_img = cv2.resize(img, (int(math.ceil(img_w * img_scaler)), int(math.ceil(img_h * img_scaler))))
    #   put the singular dimension back for grayscale images
    if img_c == 1:
        new_img = np.expand_dims(new_img, -1)
            
    #   crop (first "if" is to resize width, since height was scaler_axis; else is to resize height)
    if img_scaler_axis == 1:
        middle = int(new_img.shape[0]/2)
        start = middle - int(height/2)
        new_img = new_img[start:start+height, :, :]
    else:
        middle = int(new_img.shape[1]/2)
        start = middle - int(width/2)
        new_img = new_img[:, start:start+width, :]
        
    # stitch images together
    print("new img shape: %s" % str(new_img.shape))
    return new_img
    
####    END resize_img    ############################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputdir', type=str, required=True, help='input image directory')
    parser.add_argument('--outputdir', type=str, help='output image directory')
    parser.add_argument('--mode', type=str, default='train', help='train, val, test, or sub-directory name')
    parser.add_argument('--height', type=int, default=256, help='output image height')
    parser.add_argument('--width', type=int, default=256, help='output image width')
    parser.add_argument('--ext',type=str, default='jpg', help='image extension')
    args = parser.parse_args()

    if not args.outputdir:
        bname = os.path.basename(args.inputdir)
        dname = os.path.dirname(args.inputdir)
        args.outputdir = os.path.join(dname, '%s_resized'%bname)

    inputdir = os.path.join(args.inputdir, args.mode)
    outputdir = os.path.join(args.outputdir, args.mode)

    if not os.path.isdir(inputdir):
        print('Input directory %s not found.  Exiting with code 1.'%inputdir)
        exit(1)

    if not os.path.isdir(outputdir):
        os.makedirs(outputdir)

    imgpaths = glob(os.path.join(inputdir, '*%s'%args.ext))

    for imgpath in imgpaths:
        try:
            img = cv2.imread(imgpath)
        
            print("\n========\nImage Path: %s:" % imgpath)
            print("Image Shape: %s" % str(img.shape))
            resized_img = resize_img(img, height=args.height, width=args.width)
            newimgpath = os.path.join(outputdir, os.path.basename(imgpath))
            print("New Image Path: %s:" % newimgpath)
            
            cv2.imwrite(newimgpath, resized_img)
        
        except Exception as e:
            print("Error with %s: %s" % (imgpath, str(e)))
            print(traceback.format_exc())
