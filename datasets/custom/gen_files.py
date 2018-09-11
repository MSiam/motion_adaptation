import numpy as np
import os
import sys

img_dir= sys.argv[1]+'JPEGImages/480p/'
mask_dir= sys.argv[1]+'Annotations/480p/'
write_file= open(sys.argv[2], 'w')

def neglect_object(directory):
#   objects = ["plate", "coffe", "milk", "cereal", "jar", "medicine", "bottle5"]
#   for o in objects:
#       if o in d:
#           return True
#   return False
    if directory != 'bottle4':
        return True
    else:
        return False

for d in sorted(os.listdir(mask_dir)):
    if neglect_object(d):
        continue
    current_dir= mask_dir+d+'/'
    NSAMPLES = 0
    for f in sorted(os.listdir(current_dir)):
        if f.split('.')[1]=='png':
            NSAMPLES+= 1
#            if NSAMPLES > 60:
#                break
            f2 = f.split('_mask')[0]+'.png'
            write_file.write('JPEGImages/480p/'+d+'/'+f2 +' Annotations/480p/'+d+'/'+f+'\n')

write_file.close()

