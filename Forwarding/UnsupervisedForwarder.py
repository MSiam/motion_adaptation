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

class UnsupervisedForwarder(OneshotForwarder):
  def __init__(self, engine):
    super(UnsupervisedForwarder, self).__init__(engine)

  def _oneshot_forward_video(self, video_idx, save_logits):
    with Timer():
      # Test Network Variables + Resize output to same shape of Labels
      network = self.engine.test_network
      targets = network.raw_labels
      ys = network.y_softmax
      ys = self._adjust_results_to_targets(ys, targets)
      data = self.val_data

      # Process minibatch forward for first frame
      n, measures, ys_argmax_val, logits_val, targets_val = self._process_forward_minibatch(
        data, network, save_logits, self.save_oneshot, targets, ys, start_frame_idx=0)
      last_mask = targets_val[0]

      assert n == 1
      n_frames = data.num_examples_per_epoch()

      measures_video = []
      for t in xrange(0, n_frames):
          # Compute IoU measures
          n, measures, ys_argmax_val, posteriors_val, targets_val = self._process_forward_minibatch(
              data, network, save_logits, self.save_oneshot, targets, ys, start_frame_idx=t)
          assert n == 1
          assert len(measures) == 1
          measure = measures[0]
          print >> log.v5, "Motion Adapted frame", t, ":", measure, " factor ", float(ys_argmax_val.sum())/(854*480)

          measures_video.append(measure)

      #measures_video[:-1] = measures_video[:-1]
      measures_video = average_measures(measures_video)
      print >> log.v1, "sequence", video_idx + 1, data.video_tag(video_idx), measures_video
