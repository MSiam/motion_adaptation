#!/usr/bin/env python
'''
===============================================================================
Interactive Image Segmentation using GrabCut algorithm.

This sample shows interactive image segmentation using grabcut algorithm.

USAGE:
    python grabcut.py <filename>

README FIRST:
    Two windows will show up, one for input and one for output.

    At first, in input window, draw a rectangle around the object using
mouse right button. Then press 'n' to segment the object (once or a few times)
For any finer touch-ups, you can press any of the keys below and draw lines on
the areas you want. Then again press 'n' for updating the output.

Key '0' - To select areas of sure background
Key '1' - To select areas of sure foreground
Key '2' - To select areas of probable background
Key '3' - To select areas of probable foreground

Key 'n' - To update the segmentation
Key 'r' - To reset the setup
Key 's' - To save the results
===============================================================================
'''

# Python 2/3 compatibility
from __future__ import print_function
import os
import numpy as np
import cv2
import sys
import pdb
from scipy.ndimage.morphology import distance_transform_edt, grey_erosion

BLUE = [255,0,0]        # rectangle color
RED = [0,0,255]         # PR BG
GREEN = [0,255,0]       # PR FG
BLACK = [0,0,0]         # sure BG
WHITE = [255,255,255]   # sure FG

DRAW_BG = {'color' : BLACK, 'val' : 0}
DRAW_FG = {'color' : WHITE, 'val' : 1}
DRAW_PR_FG = {'color' : GREEN, 'val' : 3}
DRAW_PR_BG = {'color' : RED, 'val' : 2}

# setting up flags
rect = (0,0,1,1)
drawing = False         # flag for drawing curves
rectangle = False       # flag for drawing rect
rect_over = False       # flag to check if rect drawn
rect_or_mask = 100      # flag for selecting rect or mask mode
value = DRAW_FG         # drawing initialized to FG
thickness = 3           # brush thickness

def onmouse(event,x,y,flags,param):
    global img,img2,drawing,value,mask,rectangle,rect,rect_or_mask,ix,iy,rect_over

    # Draw Rectangle
    if event == cv2.EVENT_RBUTTONDOWN:
        rectangle = True
        ix,iy = x,y

    elif event == cv2.EVENT_MOUSEMOVE:
        if rectangle == True:
            img = img2.copy()
            cv2.rectangle(img,(ix,iy),(x,y),BLUE,2)
            rect = (min(ix,x),min(iy,y),abs(ix-x),abs(iy-y))
            rect_or_mask = 0

    elif event == cv2.EVENT_RBUTTONUP:
        rectangle = False
        rect_over = True
        cv2.rectangle(img,(ix,iy),(x,y),BLUE,2)
        rect = (min(ix,x),min(iy,y),abs(ix-x),abs(iy-y))
        rect_or_mask = 0
        print(" Now press the key 'n' a few times until no further change \n")

    # draw touchup curves

    if event == cv2.EVENT_LBUTTONDOWN:
        if rect_over == False:
            print("first draw rectangle \n")
        else:
            drawing = True
            cv2.circle(img,(x,y),thickness,value['color'],-1)
            cv2.circle(mask,(x,y),thickness,value['val'],-1)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.circle(img,(x,y),thickness,value['color'],-1)
            cv2.circle(mask,(x,y),thickness,value['val'],-1)

    elif event == cv2.EVENT_LBUTTONUP:
        if drawing == True:
            drawing = False
            cv2.circle(img,(x,y),thickness,value['color'],-1)
            cv2.circle(mask,(x,y),thickness,value['val'],-1)

if __name__ == '__main__':

    # print documentation
    print(__doc__)

    # Loading images
    img_dir= 'JPEGImages/480p/'
    sal_dir= 'Motion_2/'
    # input and output windows
    cv2.namedWindow('output')
    cv2.namedWindow('input')

    for d in sorted(os.listdir(sys.argv[1]+sal_dir)):
	masks= np.load(sys.argv[1]+sal_dir+d+'/sal.npy')
	indices= np.load(sys.argv[1]+sal_dir+d+'/indices.npy')
	output_list= []
	counter = 0
    	for f in sorted(os.listdir(sys.argv[1]+img_dir+d)):
	    if counter in indices:
		img = cv2.imread(sys.argv[1]+img_dir+d+'/'+f)
		img2 = img.copy()
		mask= masks[np.where(indices==counter)[0][0], :, :]
		mask_temp= np.zeros(mask.shape, dtype=np.uint8)
		mask_temp[mask>0.8]=1
		mask_temp= np.expand_dims(mask_temp, axis=2)
		eroded_mask = grey_erosion(mask_temp, size=(15, 15, 1))
		dt = distance_transform_edt(np.logical_not(eroded_mask))
		dt= dt[:,:,0]

		mask_gc=np.zeros(mask.shape, dtype=np.uint8)
		mask_gc[dt==0]= cv2.GC_FGD
		mask_gc[np.logical_and(dt>0, dt<10)]= cv2.GC_PR_FGD
		mask_gc[np.logical_and(dt>10, dt<200)]= cv2.GC_PR_BGD
		mask_gc[dt>200]= cv2.GC_BGD

		output = np.zeros(img.shape,np.uint8)           # output image to be shown
		niters= 0
		max_iters= 5
		while(niters<max_iters):
		    niters+= 1
		    cv2.imshow('output',output)
		    cv2.imshow('input',img)
		    k = cv2.waitKey(1)
		    print(""" For finer touchups, mark foreground and background after pressing keys 0-3
		    and again press 'n' \n""")
		    bgdmodel = np.zeros((1,65),np.float64)
		    fgdmodel = np.zeros((1,65),np.float64)
		    cv2.grabCut(img2,mask_gc,rect,bgdmodel,fgdmodel,1,cv2.GC_INIT_WITH_MASK)

		    mask2 = np.where((mask_gc==1) + (mask_gc==3),255,0).astype('uint8')
		    output = cv2.bitwise_and(img2,img2,mask=mask2)
		output_list.append(mask2)
 	    np.save(sys.argv[1]+sal_dir+d+'/gc.npy' , output_list)
	    counter += 1

    cv2.waitKey()
    cv2.destroyAllWindows()

