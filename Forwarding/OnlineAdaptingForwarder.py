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

class OnlineAdaptingForwarder(OneshotForwarder):
  def __init__(self, engine):
    super(OnlineAdaptingForwarder, self).__init__(engine)
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
#      files_annotations= sorted(os.listdir('/home/nray1/ms/FBMS/Annotations/480p/'+data.video_tag(video_idx) ))
      dirs= sorted(os.listdir(self.mot_dir))
      for t in xrange(0, n_frames):

          # Probability Map Function
          def get_posteriors():
              n_, _, _, logits_val_, _ = self._process_forward_minibatch(
                data, network, save_logits=False, save_results=False, targets=targets, ys=ys, start_frame_idx=t)
              assert n_ == 1
              return logits_val_[0]

          # Read adaptation target and postprocess it
          f= open(self.mot_dir+dirs[video_idx]+'/%05d.pickle'%t, 'rb')
#          f= open(self.mot_dir+data.video_tag(video_idx)+'/'+files_annotations[t].split('.')[0]+'.pickle', 'rb')
          mask = pickle.load(f)[:,:,1]
          mask= (mask- mask.min())*1.0/ (mask.max()-mask.min())
          last_mask= np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
          last_mask[mask>self.neg_th]=1
          last_mask= np.expand_dims(last_mask, axis=2)

          # Start Network Adaptation Only on first frame
          if t < self.few_shot_samples:
              negatives = self._adapt(video_idx, t, last_mask, get_posteriors, adapt_flag=1)

          # Compute IoU measures
          n, measures, ys_argmax_val, posteriors_val, targets_val = self._process_forward_minibatch(
              data, network, save_logits, self.save_oneshot, targets, ys, start_frame_idx=t)
          assert n == 1
          assert len(measures) == 1
          measure = measures[0]
          print >> log.v5, "Motion Adapted frame", t, ":", measure, " factor ", float(ys_argmax_val.sum())/(854*480)

          measures_video.append(measure)

      measures_video[:-1] = measures_video[:-1]
      measures_video = average_measures(measures_video)
      print >> log.v1, "sequence", video_idx + 1, data.video_tag(video_idx), measures_video

  def _adapt(self, video_idx, frame_idx, last_mask, get_posteriors_fn, adapt_flag=0):
    """
    adapt_flag (int): 0:do not adapt, 1:adapt with hard labels based on teacher,
                      2:adapt on hard labels from last mask, 3:adapt on soft labels
    """

    # Perform Mask erosion to reduce effect of false positive
    eroded_mask = grey_erosion(last_mask, size=(self.erosion_size, self.erosion_size, 1))

    # Compute distance transform
    dt = distance_transform_edt(numpy.logical_not(eroded_mask))

    # Adaptation target initialize
    adaptation_target = numpy.zeros_like(last_mask)
    adaptation_target[:] = VOID_LABEL

    # Retrieve current probability map to adapt with
    current_posteriors = get_posteriors_fn()
    if adapt_flag == 2:
        positives = current_posteriors[:, :, 1] > self.posterior_positive_threshold
    elif adapt_flag == 1:
        positives = last_mask==1

    if self.use_positives:
      adaptation_target[positives] = 1

    # Threshold based on distance transform
    threshold = self.distance_negative_threshold
    negatives = dt > threshold
    if self.use_negatives:
      adaptation_target[negatives] = 0

    do_adaptation = eroded_mask.sum() > 0

    # Save adaptation targets for debugging
    if self.debug:
      adaptation_target_visualization = adaptation_target.copy()
      adaptation_target_visualization[adaptation_target == 1] = 128
      if not do_adaptation:
        adaptation_target_visualization[:] = VOID_LABEL
      from scipy.misc import imsave
      folder = self.val_data.video_tag().replace("__", "/")
      imsave("forwarded/" + self.model + "/valid/" + folder + "/adaptation_%05d.png" % frame_idx,
             numpy.squeeze(adaptation_target_visualization))

    self.train_data.set_video_idx(video_idx)

    # Start Adapting based on number of adaptation_steps
    for idx in xrange(self.n_adaptation_steps):
      do_step = True
      #if idx % self.adaptation_interval == 0:
      if do_adaptation:
        feed_dict = self.train_data.feed_dict_for_video_frame(frame_idx, with_annotations=True)
        feed_dict[self.train_data.get_label_placeholder()] = adaptation_target
        loss_scale = self.adaptation_loss_scale
        adaption_frame_idx = frame_idx
      else:
        do_step = False

      if do_step:
        loss, _, n_imgs = self.trainer.train_step(epoch=idx, feed_dict=feed_dict, loss_scale=loss_scale,
                                                  learning_rate=self.adaptation_learning_rate)
        assert n_imgs == 1
        print >> log.v4, "adapting on frame", adaption_frame_idx, "of sequence", video_idx + 1, \
            self.train_data.video_tag(video_idx), "loss:", loss
    if do_adaptation:
      return negatives
    else:
      return None
