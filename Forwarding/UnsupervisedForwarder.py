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
VOID_LABEL = 128
import pickle
from PIL import Image

class UnsupervisedForwarder(OneshotForwarder):
  def __init__(self, engine):
    super(UnsupervisedForwarder, self).__init__(engine)
    self.data_dir = engine.config.unicode("davis_data_dir", "")
    if not os.path.exists(self.data_dir + 'Targets'):
        os.mkdir(self.data_dir + 'Targets')
        os.mkdir(self.data_dir + 'Targets/sq1')

  def PIL2array(self, img):
    return np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0], 4)

  def create_overlay(self, img, mask, colors):
    im= Image.fromarray(np.uint8(img))
    im= im.convert('RGBA')

    mask_color= np.zeros((mask.shape[0], mask.shape[1],3))
    if len(colors)==3:
        mask_color[mask==colors[1],0]=255
        mask_color[mask==colors[1],1]=255
        mask_color[mask==colors[2],2]=255
    else:
        mask_color[mask==colors[1],2]=255

    overlay= Image.fromarray(np.uint8(mask_color))
    overlay= overlay.convert('RGBA')

    im= Image.blend(im, overlay, 0.7)
    blended_arr= self.PIL2array(im)[:,:,:3]
    img2= img.copy()
    img2[mask!=colors[0],:]= blended_arr[mask!=colors[0],:]
    return img2

  def post_process(self, targets_val, probs_val):
    probs_val[targets_val == 1] = 0
    # Perform Mask erosion to reduce effect of false positive
    eroded_mask = grey_erosion(probs_val, size=(15, 15))
    # Compute distance transform
    dt = distance_transform_edt(numpy.logical_not(eroded_mask))
    # Adaptation target initialize
    adaptation_target = numpy.zeros(probs_val.shape, dtype=np.uint8)
    adaptation_target[:] = VOID_LABEL
    # Retrieve current probability map to adapt with
    adaptation_target[probs_val == 1] = 255
    # Threshold based on distance transform
    threshold = 60
    adaptation_target[dt > threshold] = 0
    return probs_val, adaptation_target

  def _oneshot_forward_video(self, video_idx, save_logits):
    with Timer():
      # Test Network Variables + Resize output to same shape of Labels
      network = self.engine.test_network
      targets = network.raw_labels

      ys = network.y_softmax
      ys = self._adjust_results_to_targets(ys, targets)
      data = self.val_data

      # Process minibatch forward for first frame
      n, measures, ys_argmax_val, logits_val, targets_val, fd_ = self._process_forward_minibatch(
        data, network, save_logits, self.save_oneshot, targets, ys, start_frame_idx=0)

      assert n == 1
      n_frames = data.num_examples_per_epoch()

      measures_video = []
      measures_video.append(measures[0])
      for t in xrange(0, n_frames):
          # Compute IoU measures
          n, measures, ys_argmax_val, posteriors_val, targets_val, fd = self._process_forward_minibatch(
              data, network, save_logits, self.save_oneshot, targets, ys, start_frame_idx=t)
          assert n == 1
          assert len(measures) == 1
          measure = measures[0]
          print >> log.v5, "frame", t, ":", measure

          measures_video.append(measure)

          ys_argmax_val, adap_target = self.post_process(targets_val[0, :, :, 0],
                                                         ys_argmax_val[0, :, :, 0])
          overlay = self.create_overlay(fd[fd.keys()[0]][:,:,::-1],
                                        adap_target, [0, VOID_LABEL, 255])
          cv2.imwrite(self.data_dir+"/Targets/sq1/%05d.png"%t,
                      ys_argmax_val)
          cv2.imshow('Adaptation Targets', overlay)
          cv2.waitKey(10)

      measures_video = average_measures(measures_video)
      print >> log.v1, "sequence", video_idx + 1, data.video_tag(video_idx), measures_video
