# plots loss_log.txt

import cv2, os, sys
import numpy as np
import matplotlib.pyplot as plt

project = sys.argv[1]
epochs = int(sys.argv[2])

path = "D:\\Data\\sketch_data\\checkpoints\\%s" % project
fname = os.path.join(path, "loss_log.txt")

if not os.path.exists(fname):
    print("Couldn't find log file.  Exiting")
    exit(1)
    
lossdict = {}

with open (fname, 'r') as log:
    lines = log.readlines()
    for line in lines[2:]:
        line = line.split(')')[-1]
        words = line.split()
        
        name = None
        for index, word in enumerate(words):
            
            if index % 2 == 0:
                name=word[:-1]
                if name not in lossdict.keys():
                    lossdict[name] = []
            else:
                try:
                    lossdict[name].append( float(word[:-1]) )
                except:
                    pass

plt.figure(figsize=(24, 8))
plt.ylabel("Loss")
plt.xlabel("Epoch")
for key, value in lossdict.items():
    arr = np.clip( np.array(value) , a_min=-10, a_max=10)
    x = [x * (epochs / arr.shape[0]) for x in range(arr.shape[0])]
    plt.plot(x, arr, label="%s loss"%key)
plt.legend(loc='upper left')
plt.savefig(os.path.join(path, "losses.png"))