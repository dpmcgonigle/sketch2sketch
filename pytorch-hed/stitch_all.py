#! /usr/bin/env python
# Stitch original images to sketch images

import cv2, os, sys
import numpy as np
import traceback
import math

##############################################
def stitch_imgs(img, sketch, height=256, width=256):
    """
    given input img and sketch, output an aligned imageset of [img|sketch]
    img and sketch should be of shape (c x h x w), as fetched from cv2.imread()
    """
    # get hw dims of img and sketch
    img_h, img_w, img_c = img.shape[0], img.shape[1], img.shape[2]
    sketch_h, sketch_w, sketch_c = sketch.shape[0], sketch.shape[1], sketch.shape[2]
    
    # get axis height and width scales in relation to target height and width
    img_scaler_h = height / img_h
    img_scaler_w = width / img_w
    sketch_scaler_h = height / sketch_h
    sketch_scaler_w = width / sketch_w
    
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
        
    # scale sketch based on smallest scaling ratio (ensures no whitespace when we go to crop)
    sketch_scaler_axis = 0 if sketch_scaler_h > sketch_scaler_w else 1 # scaler axis is axis that 
    sketch_scaler = sketch_scaler_h if sketch_scaler_axis == 0 else sketch_scaler_w
    new_sketch = cv2.resize(sketch, (int(math.ceil(sketch_w * sketch_scaler)), int(math.ceil(sketch_h * sketch_scaler))))
    #   put the singular dimension back for grayscale sketches
    if sketch_c == 1:
        new_sketch = np.expand_dims(new_sketch, 0)
            
    #   crop (first "if" is for height, elif for width)
    if sketch_scaler_axis == 1:
        middle = int(new_sketch.shape[0]/2)
        start = middle - int(height/2)
        new_sketch = new_sketch[start:start+height, :, :]
    else:
        middle = int(new_sketch.shape[1]/2)
        start = middle - int(width/2)
        new_sketch = new_sketch[:, start:start+width, :]   

    # stitch images together
    print("new img shape: %s" % str(new_img.shape))
    print("new sketch shape: %s" % str(new_sketch.shape))
    return np.hstack([new_img, new_sketch])
    
####    END stitch_imgs        

cwd = os.getcwd()
imgdir = os.path.join(cwd, "images")
sketchdir = os.path.join(cwd, "all-in-one", "sketch-rendered", "width-1")
stitchdir = os.path.join(cwd, "stitch_all")

if any ([not os.path.isdir(x) for x in [imgdir, sketchdir]]):
    print("Not all paths worked.  Exiting")
    exit(1)

if not os.path.isdir(stitchdir):
    os.makedirs(stitchdir)

sketches = os.listdir(sketchdir)
"""
for img in imgs[:5] + imgs[-5:]:
    print(img)
    
for sketch in sketches[:5] + sketches[-5:]:
    print (sketch)
"""
for index in range(len(sketches)):
    try:
        sketchfile = sketches[index]
        imgfile = sketchfile.split('_')[0] + ".jpg"
        img = cv2.imread(os.path.join(imgdir, imgfile))
        sketch = cv2.imread(os.path.join(sketchdir, sketchfile))
        
        print("%d:\n" % index)
        print("%s shape: %s" % (imgfile, str(img.shape)))
        print("%s shape: %s" % (sketchfile, str(sketch.shape)))
        stitched_img = stitch_imgs(img, sketch)
        
        cv2.imwrite(os.path.join(stitchdir,sketchfile), stitched_img)
        
    except Exception as e:
        print("Error with %s: %s" % (imgfile, str(e)))
        print(traceback.format_exc())
