import scipy.ndimage
import time
import numpy
import glob
import random

from Log import log
from datasets.DAVIS.DAVIS import NUM_CLASSES, VOID_LABEL, DAVIS_DEFAULT_PATH, DAVIS_FLOW_DEFAULT_PATH,\
  read_image_and_annotation_list, group_into_sequences, DAVIS_IMAGE_SIZE, DAVIS_LUCID_DEFAULT_PATH
from datasets.DAVIS.DAVIS_oneshot import DavisOneshotDataset
from datasets.Util.Util import unique_list, load_flow_from_flo
from datasets.Util.Reader import create_tensor_dict
import tensorflow as tf

class DavisContDataset(DavisOneshotDataset):
  def __init__(self, config, subset, use_old_label):
    super(DavisContDataset, self).__init__(config, subset, use_old_label)

    # override it as float32 to allow continuous labels
    self.label_placeholder = tf.placeholder(tf.float32, shape=(None, None, 1), name="label_placeholder")
