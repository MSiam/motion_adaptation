import numpy as np
import os
import sys
from shutil import copyfile

NFRAMES = 2

train_dir = '/IVOS_Translation/'
main_dir = '/IVOS_Scale/'

current_dir = 'Annotations/480p/'
current_dir2 = 'JPEGImages/480p/'
objects = os.listdir(train_dir+current_dir)
for o in objects:
    nfiles = 0
    for f in sorted(os.listdir(train_dir+current_dir+o)):
        if nfiles > NFRAMES-1:
            break
        copyfile(train_dir+current_dir+o+'/'+f, main_dir+current_dir+o+'/0'+f)
        f = f.split('_mask')[0]+'.png'
        copyfile(train_dir+current_dir2+o+'/'+f, main_dir+current_dir2+o+'/0'+f)
        print(train_dir+current_dir+o+'/'+f)
        nfiles += 1
