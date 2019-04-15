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

class TeacherAdaptingForwarder(OneshotForwarder):
  def __init__(self, engine):
    super(TeacherAdaptingForwarder, self).__init__(engine)
    self.n_adaptation_steps = self.config.int("n_adaptation_steps", 12)
    self.adaptation_interval = self.config.int("adaptation_interval", 4)
    self.adaptation_learning_rate = self.config.float("adaptation_learning_rate")
    self.posterior_positive_threshold = self.config.float("posterior_positive_threshold", 0.97)
    self.distance_negative_threshold = self.config.float("distance_negative_threshold", 150.0)
    self.adaptation_loss_scale = self.config.float("adaptation_loss_scale", 0.1)
    self.debug = self.config.bool("adapt_debug", False)
    self.erosion_size = self.config.int("adaptation_erosion_size", 20)
    self.use_positives = self.config.bool("use_positives", True)
    self.use_negatives = self.config.bool("use_negatives", True)
    self.mot_dir= self.config.unicode("targets_path", "")
    self.neg_th = self.config.float("adapt_th", 0.8)
    self.few_shot_samples = self.config.int("few_shot_samples", 1)
    self.dataset = self.config.unicode("davis_data_dir", "")
    self.adapt_flag = 1
    self.rescale = self.config.float("davis_img_size_rescale")

  def _oneshot_forward_video(self, video_idx, save_logits):
    with Timer():
      # Test Network Variables + Resize output to same shape of Labels
      network = self.engine.test_network
      targets = network.raw_labels
      ys = network.y_softmax
      ys = self._adjust_results_to_targets(ys, targets)
      data = self.val_data

      # Probability Map Function
      def get_posteriors():
          n_, _, _, logits_val_, _ = self._process_forward_minibatch(
            data, network, save_logits=False, save_results=False, targets=targets, ys=ys, start_frame_idx=t)
          assert n_ == 1
          return logits_val_[0]

      # Adapt the network based on number of few_shot_samples
      files_annotations = sorted(os.listdir(self.mot_dir + 'sq1/'))
      for t in range(self.few_shot_samples):
          f= self.mot_dir + 'sq1/' + files_annotations[t]
          mask = cv2.imread(f, 0)
          last_mask= np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
          last_mask[mask==255] = 1
          last_mask[mask==128] = VOID_LABEL
          new_shape = (int(self.rescale * last_mask.shape[1]), int(self.rescale * last_mask.shape[0]))
          last_mask = cv2.resize(last_mask, new_shape, cv2.INTER_NEAREST)
          last_mask= np.expand_dims(last_mask, axis=2)
          self._adapt(0, t, last_mask, get_posteriors)
          print >> log.v5, "Motion Adapted frame", t, ":", measure

      print("Finished Adaptation")
      # Infer on Camera Live Feed Input
#      while True:
#
#          # Compute IoU measures
#          _, _, ys_argmax_val, posteriors_val, _ = self._process_forward_minibatch(
#              data, network, save_logits, self.save_oneshot, targets, ys, start_frame_idx=t)


  def _adapt(self, video_idx, frame_idx, last_mask, get_posteriors_fn):
    """
    adapt_flag (int): 0:do not adapt, 1:adapt with hard labels based on teacher,
                      2:adapt on hard labels from last mask, 3:use continuous labels
    """
    adaptation_target = last_mask
    self.train_data.set_video_idx(video_idx)

    # Start Adapting based on number of adaptation_steps
    for idx in xrange(self.n_adaptation_steps):
      feed_dict = self.train_data.feed_dict_for_video_frame(frame_idx, with_annotations=True)
      feed_dict[self.train_data.get_label_placeholder()] = adaptation_target
      loss_scale = self.adaptation_loss_scale
      adaption_frame_idx = frame_idx

      loss, _, n_imgs = self.trainer.train_step(epoch=idx, feed_dict=feed_dict, loss_scale=loss_scale,
                                                learning_rate=self.adaptation_learning_rate)
      print >> log.v4, "adapting on frame", adaption_frame_idx, "of sequence", video_idx + 1, \
               self.train_data.video_tag(video_idx), "loss:", loss
