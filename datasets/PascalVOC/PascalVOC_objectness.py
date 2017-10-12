from datasets.PascalVOC.PascalVOC import PascalVOCDataset, VOID_LABEL

import tensorflow as tf
import numpy

NUM_CLASSES = 2


def postproc_labels_for_objectness(label):
  # 0 is background
  # 255 is void
  # 1-20 are normal classes

  def my_postproc(l):
    l_out = numpy.zeros_like(l)
    l_out[numpy.logical_and(l != 0, l != 255)] = 1
    #do we want void here?
    l_out[l == 255] = VOID_LABEL
    return l_out

  label_out, = tf.py_func(my_postproc, [label], [label.dtype])
  label_out.set_shape(label.get_shape())
  return label_out


class PascalVOCObjectnessDataset(PascalVOCDataset):
  def __init__(self, config, subset, coord, fraction=1.0):
    super(PascalVOCObjectnessDataset, self).__init__(config, subset, coord,
                                                     label_postproc_fn=postproc_labels_for_objectness,
                                                     name="pascalvoc_objectness",
                                                     num_classes=NUM_CLASSES, fraction=fraction, ignore_classes=[0])
