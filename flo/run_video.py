import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco
import cv2

def segment_person(masks, classes, class_names):
    person_mask = np.zeros_like(masks[:, :, 0])

    for i, c in zip(range(masks.shape[2]), classes):
        if class_names[c] == 'person':
            person_mask[masks[:, :, i] == 1] = 1
    return person_mask

def PIL2array(img):
    return np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0], 4)

def create_overlay(img, mask, colors):
    im= Image.fromarray(np.uint8(img))
    im= im.convert('RGBA')
    mask_color= np.zeros((mask.shape[0], mask.shape[1],3))

    if len(colors)==3:
        mask_color[mask==colors[1],0]=255
        mask_color[mask==colors[1],1]=255
        mask_color[mask==colors[2],0]=255
    else:
        mask_color[mask==colors[1],2]=255

    overlay= Image.fromarray(np.uint8(mask_color))
    overlay= overlay.convert('RGBA')

    im= Image.blend(im, overlay, 0.7)
    blended_arr= PIL2array(im)[:,:,:3]
    img2= img.copy()
    img2[mask==colors[1],:]= blended_arr[mask==colors[1],:]
    return img2

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = sys.argv[1]#os.path.join(ROOT_DIR, "images")

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

# Load a random image from the images folder
#for d in sorted(os.listdir(IMAGE_DIR + 'JPEGImages/')):
dirs = sorted(os.listdir(IMAGE_DIR + 'JPEGImages/'))
d = dirs[int(sys.argv[2])]
file_names = sorted(os.listdir(IMAGE_DIR + 'JPEGImages/'+d+'/'))
if not os.path.exists(IMAGE_DIR + 'Annotations'):
    os.mkdir(IMAGE_DIR + 'Annotations')
    os.mkdir(IMAGE_DIR + 'Annotations/'+d+'/')

counter = 0
for idx in range(len(file_names)):
    image = skimage.io.imread(os.path.join(IMAGE_DIR + 'JPEGImages/'+d+'/', file_names[idx]))
    results = model.detect([image], verbose=1)

    masks = results[0]['masks']
    classes = results[0]['class_ids']
    person_mask = segment_person(masks, classes, class_names)
    overlay = create_overlay(image[:,:,::-1], person_mask, [0, 1])
    cv2.imshow('Segment Human', overlay)
    cv2.waitKey(10)
    cv2.imwrite(IMAGE_DIR + '/Annotations/'+d+'/%05d.png'%counter, person_mask*255)
    counter += 1
