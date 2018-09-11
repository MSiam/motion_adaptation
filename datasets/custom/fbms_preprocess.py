import numpy as np
import cv2
import os

main_dir = '/home/nray1/ms/FBMS/Annotations/480p/'
temp_dir = '/home/nray1/ms/FBMS/Annotations2/480p/'

for d in sorted(os.listdir(main_dir)):
    for f in sorted(os.listdir(main_dir+d)):
        mask = cv2.imread(main_dir+d+'/'+f, 0)
        mask_temp = np.zeros_like(mask)
        mask_temp[mask == 0] = 255

        cv2.imwrite(temp_dir+d+'/'+f, mask_temp)
