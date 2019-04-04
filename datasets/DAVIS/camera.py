import scipy.ndimage
import time
import numpy
import glob
import random

from datasets.DAVIS.DAVIS import NUM_CLASSES, VOID_LABEL, DAVIS_IMAGE_SIZE
from datasets.FeedDataset import OneshotImageDataset
import cv2
import numpy as np
import os

class CameraOneShot(OneshotImageDataset):
  def __init__(self, config, subset, use_old_label):
    self.flow_into_past = config.bool("flow_into_past", False)
    self.flow_into_future = config.bool("flow_into_future", False)
    self.twostream = config.bool("twostream", False)
    self.data_dir = config.unicode("data_dir", '')

    super(CameraOneShot, self).__init__(config, NUM_CLASSES, VOID_LABEL, subset, image_size=DAVIS_IMAGE_SIZE,
                                              use_old_label=use_old_label, flow_into_past=self.flow_into_past,
                                              flow_into_future=self.flow_into_future)
    self.config = config
    self.current_idx = 0
    self.load_frames()

  def load_frames(self):
      files = sorted(os.listdir(self.data_dir))
      self.video = []
      for f in files:
        if 'img' in f:
            img = cv2.imread(self.data_dir + f)
            flow = cv2.imread(self.data_dir + f.replace('img', 'flo'))
            self.video.append((img, flow))
      self.img_shape = self.video[0][0].shape

  def _get_video_data(self):
    img_pair = self.video[self.current_idx]
    tensors = {"unnormalized_img": img_pair[0] / 255.0, "tag": "camera",
                "flow": img_pair[1],
               "label": np.zeros((self.img_shape[0], self.img_shape[1], 1))}

    return [tensors]
