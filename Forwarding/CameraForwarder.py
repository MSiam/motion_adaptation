from Forwarding.OneshotForwarder import OneshotForwarder
from datasets.Util.Timer import Timer
from Measures import average_measures
from Log import log
import os
import numpy
from scipy.ndimage.morphology import distance_transform_edt, grey_erosion
import scipy.misc
import numpy as np
import cv2
VOID_LABEL = 255
import pickle
import cv2
import matplotlib.pyplot as plt
import numpy as np
#from flownet2tf.src.net import Mode
#from flownet2tf.src.flownet2.flownet2 import FlowNet2
import tensorflow as tf
import subprocess

class CameraForwarder(OneshotForwarder):
  def __init__(self, engine):
    super(CameraForwarder, self).__init__(engine)

  def forward(self, network, data, save_results=True, save_logits=False):
      network = self.engine.test_network
      data = self.val_data
      ys = network.y_softmax
      targets = None

      # Process minibatch forward for first frame
      for i in range(len(data.video)):
          data.current_idx = i
          n, measures, ys_argmax_val, posteriors_val, targets_val = self._process_forward_minibatch(
              data, network, save_logits, self.save_oneshot, targets, ys, start_frame_idx=0)
          preds = np.asarray(ys_argmax_val[0, :, :, 0], dtype=np.float32)
          preds = cv2.resize(preds, (data.img_shape[1], data.img_shape[0]))

          cv2.imshow('Predictions', preds)
          print('processed frame ', i)
          ch = cv2.waitKey(10)%256
          if ch == ord('q'): # quit
              break


