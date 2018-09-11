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

class TeacherContAdaptingForwarder(OneshotForwarder):
  def __init__(self, engine):
    super(TeacherContAdaptingForwarder, self).__init__(engine)
    self.n_adaptation_steps = self.config.int("n_adaptation_steps", 12)
    self.adaptation_interval = self.config.int("adaptation_interval", 4)
    self.adaptation_learning_rate = self.config.float("adaptation_learning_rate")
    self.adaptation_loss_scale = self.config.float("adaptation_loss_scale", 0.1)
    self.debug = self.config.bool("adapt_debug", False)
    self.use_positives = self.config.bool("use_positives", True)
    self.use_negatives = self.config.bool("use_negatives", True)
    self.mot_dir= self.config.unicode("targets_path", "")
    self.few_shot_samples = self.config.int("few_shot_samples", 1)
    self.dataset = self.config.unicode("davis_data_dir", "")

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
      measures_video.append(measures[0])
      dirs= sorted(os.listdir(self.mot_dir))
      files_annotations = sorted(os.listdir(self.mot_dir+data.video_tag(video_idx)))
#      if "FBMS" in self.dataset:
#          files_annotations= sorted(os.listdir('/home/nray1/ms/FBMS/Annotations/480p/'+data.video_tag(video_idx) ))
#      elif "FORDS_Rotation" in self.dataset or \
#              "PDB" in self.mot_dir or \
#              "FORDS_tasks" in self.dataset:
#          files_annotations = sorted(os.listdir(self.mot_dir+data.video_tag(video_idx)))
#      elif "FORD" in self.dataset:
#          files_annotations = sorted(os.listdir(self.mot_dir+dirs[video_idx]))
      for t in xrange(0, n_frames):

          # Probability Map Function
          def get_posteriors():
              n_, _, _, logits_val_, _ = self._process_forward_minibatch(
                data, network, save_logits=False, save_results=False, targets=targets, ys=ys, start_frame_idx=t)
              assert n_ == 1
              return logits_val_[0]

          # Start Network Adaptation Only on first frame
          if t < self.few_shot_samples:
              # Read adaptation target and postprocess it
              # For DAVIS starts at 0, FORDS starts at 1 for frame numbers, FBMS use annotation files
              if "FBMS" in self.mot_dir:
                  mask = cv2.imread(self.mot_dir+data.video_tag(video_idx)+'/'+files_annotations[t], 0)
              else:
                  if "FORDS" in self.dataset:
                      f= open(self.mot_dir+data.video_tag(video_idx)+'/'+files_annotations[t], 'rb')
#                  elif "FORDS_Rotation" in self.dataset:
#                      f= open(self.mot_dir+data.video_tag(video_idx)+'/'+files_annotations[t], 'rb')
#                  elif "FORD" in self.dataset:
#                      f= open(self.mot_dir+dirs[video_idx]+'/'+files_annotations[t], 'rb')
                  elif "DAVIS" in self.dataset:
                     f= open(self.mot_dir+dirs[video_idx]+'/%05d.pickle'%(t), 'rb')
#                  else:
#                     f= open(self.mot_dir+data.video_tag(video_idx)+'/'+files_annotations[t].split('.')[0]+'.pickle', 'rb')
                  mask = pickle.load(f)[:,:,1]
              mask= (mask- mask.min())*1.0/ (mask.max()-mask.min())
              last_mask= np.expand_dims(mask, axis=2)

              self._adapt(video_idx, t, last_mask, get_posteriors)

          # Compute IoU measures
          n, measures, ys_argmax_val, posteriors_val, targets_val = self._process_forward_minibatch(
              data, network, save_logits, self.save_oneshot, targets, ys, start_frame_idx=t)
          assert n == 1
          assert len(measures) == 1
          measure = measures[0]
          print >> log.v5, "Motion Adapted frame", t, ":", measure
          measures_video.append(measure)

      measures_video = average_measures(measures_video)
      print >> log.v1, "sequence", video_idx + 1, data.video_tag(video_idx), measures_video

  def _adapt(self, video_idx, frame_idx, last_mask, get_posteriors_fn):
    adaptation_target = last_mask
    do_adaptation = adaptation_target.sum() > 0

    # Save adaptation targets for debugging
    if self.debug:
      adaptation_target_visualization = adaptation_target.copy()
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
