#! /usr/bin/env python

"""
Reduced aligned image dataset to only one side.

Ex: ./unstitch.py --datadir val_targets --side B --ext png --width 256
"""

import cv2
import os, sys
import numpy as np
import argparse
from glob import glob

parser = argparse.ArgumentParser()
parser.add_argument('--datadir', type=str, required=True, help='Directory of dataset to split')
parser.add_argument('--side', type=str, default='A', help='Which side to keep')
parser.add_argument('--ext', type=str, default='png', help='Image extension')
parser.add_argument('--width', type=int, default=256, help='Image width')
args = parser.parse_args()

imagepaths = glob( os.path.join(args.datadir, '*.%s'%args.ext) )

for imagepath in imagepaths:
    print('Image path: %s' % imagepath)
    image = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)
    newimg = image[:, :args.width] if args.side == 'A' else image [:, args.width:]
    cv2.imwrite(imagepath, newimg)
