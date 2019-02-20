import numpy as np
import os
import sys
from shutil import copyfile

current_dir = '/IVOS/objects_full/daylight/rotation/'
out_dir = '/IVOS_Rotation/JPEGImages/480p/'

objects = os.listdir(current_dir)
for o in objects:
    files = sorted(os.listdir(current_dir+o))
    for f in files:
        if not os.path.exists(out_dir+o):
            os.mkdir(out_dir+o)

        copyfile(current_dir+o+'/'+f, out_dir+o+'/'+f)
