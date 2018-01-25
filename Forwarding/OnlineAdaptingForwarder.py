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
#import pdb
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
    self.mot_dir= '/home/eren/Data/DAVIS/Motion/'
    self.short_dir= '/home/eren/Data/FBMS/Testset/Motion_4/'
#    self.short_dir= '/home/eren/Work/MTLMotion/forwarded/DAVIS16_oneshot/valid/'
#    self.short_dir= '/home/eren/Data/DAVIS/Motion_4/'
#    self.long_dir= '/home/eren/Data/DAVIS/ARP/'
    self.neg_th = 0.8

  def _oneshot_forward_video(self, video_idx, save_logits):
    with Timer():
      # finetune on first frame
      self._finetune(video_idx, n_finetune_steps=self.n_finetune_steps)

      network = self.engine.test_network
      targets = network.raw_labels
      ys = network.y_softmax
      ys = self._adjust_results_to_targets(ys, targets)
      data = self.val_data

      n, measures, ys_argmax_val, logits_val, targets_val = self._process_forward_minibatch(
        data, network, save_logits, self.save_oneshot, targets, ys, start_frame_idx=0)
      last_mask = targets_val[0]

      adapt_flag= True
      assert n == 1
      n_frames = data.num_examples_per_epoch()

      measures_video = []

      #dirs= sorted(os.listdir(self.mot_dir))
      dirs= os.listdir(self.short_dir)
      #motype= np.load(self.mot_dir+'/'+dirs[video_idx]+'/motype.npy')
      motype= 'moving'
      print('Motion Type of this Sequence is ', motype)

#      masks= np.load(self.mot_dir+dirs[video_idx]+'/mask_'+dirs[video_idx]+'.npy')
#      indices= np.load(self.mot_dir+dirs[video_idx]+'/indices.npy')
      files_motion= []#sorted(os.listdir(self.short_dir+dirs[video_idx]))
      files_annotations= sorted(os.listdir('/home/eren/Data/FBMS/Testset/Annotations/'+dirs[video_idx]))
      for f in files_annotations:
          if f.split('.')[1]=='png':
            if '_gt' in f:
                f=f.split('_gt')[0]+'.png'
            files_motion.append(f)
      #pdb.set_trace()

      for t in xrange(0, n_frames):
          def get_posteriors():
              n_, _, _, logits_val_, _ = self._process_forward_minibatch(
                data, network, save_logits=False, save_results=False, targets=targets, ys=ys, start_frame_idx=t)
              assert n_ == 1
              return logits_val_[0]
          if motype=='static':
              temp= cv2.imread(self.long_dir+dirs[video_idx]+('/%05d.png'%t), 0)
              temp= (temp- temp.min())*1.0/ (temp.max()-temp.min())
              last_mask= np.zeros((temp.shape[0], temp.shape[1]), dtype=np.uint8)
              last_mask[temp>0.5]=1
              last_mask= np.expand_dims(last_mask, axis=2)
              self.distance_negative_threshold= 10
              if adapt_flag:
                  negatives = self._adapt(video_idx, t, last_mask, get_posteriors, adapt_flag=1)
     #             adapt_flag= False
              n, measures, ys_argmax_val, posteriors_val, targets_val = self._process_forward_minibatch(
                  data, network, save_logits, self.save_oneshot, targets, ys, start_frame_idx=t)
              assert n == 1
              assert len(measures) == 1
              measure = measures[0]
              print >> log.v5, "Motion Adapted frame", t, ":", measure, " factor ", float(ys_argmax_val.sum())/(ys_argmax_val.shape[1]*ys_argmax_val.shape[2])
          else:
              if t<n_frames-1:
                  if adapt_flag:
                      #ff= open(self.short_dir+dirs[video_idx]+'/'+files_motion[t*2])
                      #temp= pickle.load(ff)[:,:,1]
                      temp= cv2.imread(self.short_dir+dirs[video_idx]+'/'+files_motion[t], 0)
                      temp= (temp- temp.min())*1.0/ (temp.max()-temp.min())
                      last_mask= np.zeros((temp.shape[0], temp.shape[1]), dtype=np.uint8)
                      last_mask[temp>self.neg_th]=1
                      if last_mask.sum()!=0:
                          last_mask= np.expand_dims(last_mask, axis=2)
                          negatives = self._adapt(video_idx, t, last_mask, get_posteriors, adapt_flag=1)
       #                   adapt_flag= False

              n, measures, ys_argmax_val, posteriors_val, targets_val = self._process_forward_minibatch(
                  data, network, save_logits, self.save_oneshot, targets, ys, start_frame_idx=t)
              assert n == 1
              assert len(measures) == 1
              measure = measures[0]
              print >> log.v5, "Motion Adapted frame", t, ":", measure, " factor ", float(ys_argmax_val.sum())/(ys_argmax_val.shape[1]*ys_argmax_val.shape[2])#(854*480)

          measures_video.append(measure)
          #last_mask = ys_argmax_val[0]

      # prune negatives from last mask
      # negatives are None if we think that the target is lost
#      if negatives is not None and self.use_negatives:
#        last_mask[negatives] = 0
      #########

      measures_video = measures_video[1:-1]
      measures_video = average_measures(measures_video)
      print >> log.v1, "sequence", video_idx + 1, data.video_tag(video_idx), measures_video

  def _adapt(self, video_idx, frame_idx, last_mask, get_posteriors_fn, adapt_flag=0):
    eroded_mask = grey_erosion(last_mask, size=(self.erosion_size, self.erosion_size, 1))
    dt = distance_transform_edt(numpy.logical_not(eroded_mask))

    adaptation_target = numpy.zeros_like(last_mask)
    adaptation_target[:] = VOID_LABEL

    current_posteriors = get_posteriors_fn()
    if adapt_flag==0:
        positives = current_posteriors[:, :, 1] > self.posterior_positive_threshold
    else:
        positives = last_mask==1

    if self.use_positives:
      adaptation_target[positives] = 1

    threshold = self.distance_negative_threshold
    negatives = dt > threshold
    if self.use_negatives:
      adaptation_target[negatives] = 0

    do_adaptation = eroded_mask.sum() > 0

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

    for idx in xrange(self.n_adaptation_steps):
      do_step = True
      #if idx % self.adaptation_interval == 0:
      if do_adaptation:
        feed_dict = self.train_data.feed_dict_for_video_frame(frame_idx, with_annotations=True)
        feed_dict[self.train_data.get_label_placeholder()] = adaptation_target
        loss_scale = self.adaptation_loss_scale
        adaption_frame_idx = frame_idx
      else:
        print >> log.v4, "skipping current frame adaptation, since the target seems to be lost"
        do_step = False
      #else:
        # mix in first frame to avoid drift
        # (do this even if we think the target is lost, since then this can help to find back the target)
      #  feed_dict = self.train_data.feed_dict_for_video_frame(frame_idx=0, with_annotations=True)
      #  loss_scale = 1.0
      #  adaption_frame_idx = 0

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
