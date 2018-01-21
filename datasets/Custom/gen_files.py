import numpy as np
import os
import sys
import pdb

img_dir= sys.argv[1]+'JPEGImages/'
mask_dir= sys.argv[1]+'Annotations/'
write_file= open(sys.argv[2], 'w')

for d in os.listdir(mask_dir):
    current_dir= mask_dir+d+'/'
    for f in os.listdir(current_dir):
        if f.split('.')[1]=='png':
            if '_gt' in f:
                write_file.write('JPEGImages/'+d+'/'+f.split('_gt')[0]+'.jpg '+'Annotations/'+d+'/'+f+'\n')
            else:
                write_file.write('JPEGImages/'+d+'/'+f.split('.')[0]+'.jpg '+'Annotations/'+d+'/'+f+'\n')

write_file.close()
