import Measures
from Forwarding.Forwarder import ImageForwarder
import time
from math import ceil
from Log import log


class BaseForwarder(ImageForwarder):
  """
  Baseline Forwarder with no adaptation
  """
  def __init__(self, engine):
    super(BaseForwarder, self).__init__(engine)
    self.val_data = self.engine.valid_data
    self.forward_interval = self.config.int("forward_interval", 9999999)
    self.forward_initial = self.config.bool("forward_initial", False)
    self.video_range = self.config.int_list("video_range", [])
    self.video_ids = self.config.int_list("video_ids", [])
    assert len(self.video_range) == 0 or len(self.video_ids) == 0, "cannot specify both"
    self.save_baseline = self.config.bool("save_baseline", False)

  def forward(self, network, data, save_results=True, save_logits=False):
    if len(self.video_range) != 0:
      video_ids = range(self.video_range[0], self.video_range[1])
    elif len(self.video_ids) != 0:
      video_ids = self.video_ids
    else:
      video_ids= [self.config.vid]
      print('specific video_ids ', video_ids)

    for video_idx in video_ids:
      tag = self.val_data.video_tag(video_idx)
      print >> log.v4, "finetuning on", tag

      # reset weights and optimizer for next video
      self.engine.try_load_weights()
      self.engine.reset_optimizer()
      self.val_data.set_video_idx(video_idx)

      print >> log.v4, "steps:", 0
      self.forward_video(video_idx, save_logits)

  def forward_video(self, video_idx, save_logits):
    forward_interval = self.forward_interval
    save_results = self.save_baseline# and i == n_partitions - 1
    save_logits_here = save_logits #and i == n_partitions - 1
    self._base_forward(self.engine.test_network, self.val_data, save_results=save_results,
                         save_logits=save_logits_here)

  def _base_forward(self, network, data, save_results, save_logits):
    super(BaseForwarder, self).forward(self.engine.test_network, self.val_data, save_results, save_logits)
