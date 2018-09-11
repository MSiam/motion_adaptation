import numpy as np
import os
import sys
import cv2
import cPickle
import pickle

img_dir= sys.argv[1]+'JPEGImages/480p/'
mask_dir= sys.argv[1]+'Annotations/480p/'

for d in sorted(os.listdir(mask_dir)):
    if "plate" in d or "coffe" in d or "milk" in d or "cereal" in d:
        continue
    current_dir= mask_dir+d+'/'
    NSAMPLES = 0
    for f in sorted(os.listdir(current_dir)):
        if f.split('.')[1]=='png':
            NSAMPLES+= 1
            print('file ', mask_dir+d+'/'+f)
            mask_img = cv2.imread(mask_dir+d+'/'+f, 0)
            mask_img_res = np.zeros((480, mask_img.shape[1]), dtype=mask_img.dtype)
            mask_img_res[:mask_img.shape[0], :] = mask_img
            cv2.imwrite(mask_dir+d+'/'+f, mask_img_res)

            f2 = f.split('_mask')[0]+'.png'
            print('file ', img_dir+d+'/'+f2)
            img = cv2.imread(img_dir+d+'/'+f2)
            img_res = np.zeros((480, img.shape[1], 3), dtype=img.dtype)
            img_res[:img.shape[0], :, :] = img
            cv2.imwrite(img_dir+d+'/'+f2, img_res)

#mask_dir= '/home/nray1/ms/2stream_motion_adaptation/MTLMotion/forwarded/targets_filtered/'
#
#for d in sorted(os.listdir(mask_dir)):
#    if "plate" in d or "coffe" in d or "milk" in d or "cereal" in d:
#        continue
#    current_dir= mask_dir+d+'/'
#    NSAMPLES = 0
#    for f in sorted(os.listdir(current_dir)):
#        if f.split('.')[1]=='pickle':
#            mask_f = open(mask_dir+d+'/'+f, 'rb')
#            mask = pickle.load(mask_f)
#            mask_res = np.zeros((480, mask.shape[1], 2), dtype=mask.dtype)
#            mask_res[:mask.shape[0], :, :] = mask
#            cPickle.dump(mask_res, open(mask_dir+d+'/'+f, "w"), cPickle.HIGHEST_PROTOCOL)
