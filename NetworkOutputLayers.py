import numpy
import tensorflow as tf

import Constants
from Measures import create_confusion_matrix, get_average_precision, compute_binary_ious_tf
from NetworkLayers import Layer, L2_DEFAULT, BATCH_NORM_DECAY_DEFAULT
from Util_Network import prepare_input, global_avg_pool, prepare_collapsed_input_and_dropout, get_activation, \
  apply_dropout, conv2d, conv2d_dilated
from datasets.Util.Util import smart_shape

MAX_ADJUSTABLE_CLASSES = 100  # max 100 objects per sequence should be sufficient


class Softmax(Layer):
  output_layer = True

  def __init__(self, name, inputs, targets, n_classes, tower_setup, global_average_pooling=False, dropout=0.0,
               loss="ce", l2=L2_DEFAULT):
    super(Softmax, self).__init__()
    self.measures = {}
    if global_average_pooling:
      inp, n_features_inp = prepare_input(inputs)
      inp = global_avg_pool(inp)
    else:
      inp, n_features_inp = prepare_collapsed_input_and_dropout(inputs, dropout)

    with tf.variable_scope(name):
      W = self.create_weight_variable("W", [n_features_inp, n_classes], l2, tower_setup)
      b = self.create_bias_variable("b", [n_classes], tower_setup)
      y_ref = tf.cast(targets, tf.int64)
      y_pred = tf.matmul(inp, W) + b
      self.outputs = [tf.nn.softmax(y_pred, -1, 'softmax')]
      errors = tf.not_equal(tf.argmax(y_pred, 1), y_ref)
      errors = tf.reduce_sum(tf.cast(errors, tower_setup.dtype))
      self.measures['errors'] = errors

      if loss == "ce":
        cross_entropy_per_example = tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=y_pred, labels=y_ref, name='cross_entropy_per_example')
        self.loss = tf.reduce_sum(cross_entropy_per_example, name='cross_entropy_sum')
      else:
        assert False, "Unknown loss " + loss

      self.add_scalar_summary(self.loss, "loss")


def bootstrapped_ce_loss(ce, fraction):
  # only consider k worst pixels (lowest posterior probability) per image
  assert fraction is not None
  batch_size = ce.get_shape().as_list()[0]
  if batch_size is None:
    batch_size = tf.shape(ce)[0]
  k = tf.cast(tf.cast(tf.shape(ce)[1] * tf.shape(ce)[2], tf.float32) * fraction, tf.int32)
  bs_ce, _ = tf.nn.top_k(tf.reshape(ce, shape=[batch_size, -1]), k=k, sorted=False)
  bs_ce = tf.reduce_mean(bs_ce, axis=1)
  bs_ce = tf.reduce_sum(bs_ce, axis=0)
  return bs_ce


class SegmentationSoftmax(Layer):
  output_layer = True

  def create_weights(self, n_classes, filter_size, n_features_inp, l2, tower_setup):
    if n_classes is None:
      n_class_weights = 2
    else:
      n_class_weights = n_classes
    W = self.create_weight_variable("W", filter_size + [n_features_inp, n_class_weights], l2, tower_setup)
    b = self.create_bias_variable("b", [n_class_weights], tower_setup)

    W_used = W
    b_used = b
    if n_classes is None:
      with tf.device(tower_setup.variable_device):
        n_classes_current = tf.get_variable("n_classes_current", shape=[], trainable=False, dtype=tf.int32)
        W_adjustable = tf.get_variable("W_adjustable", filter_size + [n_features_inp, MAX_ADJUSTABLE_CLASSES])
        b_adjustable = tf.get_variable("b_adjustable", [MAX_ADJUSTABLE_CLASSES])
        if l2 > 0.0:
          self.regularizers.append(l2 * tf.nn.l2_loss(W_adjustable))
        W_used = W_adjustable[..., :n_classes_current]
        b_used = b_adjustable[:n_classes_current]
    else:
      W_adjustable = b_adjustable = n_classes_current = None
    return W, b, W_adjustable, b_adjustable, n_classes_current, W_used, b_used

  def _create_adjustable_output_assign_data(self, tower_setup):
    if self.W_adjustable is None or self.b_adjustable is None:
      return None
    else:
      W_adjustable_val_placeholder = tf.placeholder(tower_setup.dtype, name="W_adjustable_val_placeholder")
      b_adjustable_val_placeholder = tf.placeholder(tower_setup.dtype, name="b_adjustable_val_placeholder")
      n_classes_current_val_placeholder = tf.placeholder(tf.int32, name="n_classes_current_val_placeholder")
      assign_W_adjustable = tf.assign(self.W_adjustable, W_adjustable_val_placeholder)
      assign_b_adjustable = tf.assign(self.b_adjustable, b_adjustable_val_placeholder)
      assign_n_classes_current = tf.assign(self.n_classes_current, n_classes_current_val_placeholder)
      return assign_W_adjustable, assign_b_adjustable, assign_n_classes_current, W_adjustable_val_placeholder, \
          b_adjustable_val_placeholder, n_classes_current_val_placeholder

  def adjust_weights_for_multiple_objects(self, session, n_objects):
    W_val, b_val = session.run([self.W, self.b])
    W_adjustable_val_new = numpy.zeros(W_val.shape[:-1] + (MAX_ADJUSTABLE_CLASSES,), dtype="float32")
    b_adjustable_val_new = numpy.zeros(MAX_ADJUSTABLE_CLASSES, dtype="float32")

    W_adjustable_val_new[..., :n_objects + 1] = W_val[..., [0] + ([1] * n_objects)]
    b_adjustable_val_new[:n_objects + 1] = b_val[[0] + ([1] * n_objects)]
    b_adjustable_val_new[1:n_objects + 1] -= numpy.log(n_objects)

    assign_W_adjustable, assign_b_adjustable, assign_n_classes_current, W_adjustable_val_placeholder, \
        b_adjustable_val_placeholder, n_classes_current_val_placeholder = self.adjustable_output_assign_data

    session.run([assign_W_adjustable, assign_b_adjustable, assign_n_classes_current],
                feed_dict={W_adjustable_val_placeholder: W_adjustable_val_new,
                           b_adjustable_val_placeholder: b_adjustable_val_new,
                           n_classes_current_val_placeholder: n_objects + 1})

  @staticmethod
  def create_loss(loss_str, fraction, no_void_label_mask, targets, tower_setup, void_label, y_pred):
    ce = None
    if "ce" in loss_str:
      ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred, labels=targets, name="ce")

      if void_label is not None:
        mask = tf.cast(no_void_label_mask, tower_setup.dtype)
        ce *= mask
    if loss_str == "ce":
      ce = tf.reduce_mean(ce, axis=[1, 2])
      ce = tf.reduce_sum(ce, axis=0)
      loss = ce
    elif loss_str == "bootstrapped_ce":
      bs_ce = bootstrapped_ce_loss(ce, fraction)
      loss = bs_ce
    else:
      assert False, "Unknown loss " + loss_str
    return loss

  def create_measures(self, n_classes, pred, targets):
    measures = {}
    conf_matrix = tf.py_func(create_confusion_matrix, [pred, targets, self.n_classes_current], [tf.int64])
    measures[Constants.CONFUSION_MATRIX] = conf_matrix[0]
    return measures

  def __init__(self, name, inputs, targets, n_classes, void_label, tower_setup, filter_size=(1, 1),
               input_activation=None, dilation=None, resize_targets=False, resize_logits=False, loss="ce",
               fraction=None, l2=L2_DEFAULT, dropout=0.0):
    super(SegmentationSoftmax, self).__init__()
    assert targets.get_shape().ndims == 4, targets.get_shape()
    assert not (resize_targets and resize_logits)
    inp, n_features_inp = prepare_input(inputs)

    filter_size = list(filter_size)

    with tf.variable_scope(name):
      if input_activation is not None:
        inp = get_activation(input_activation)(inp)

      inp = apply_dropout(inp, dropout)

      self.W, self.b, self.W_adjustable, self.b_adjustable, self.n_classes_current, W, b = self.create_weights(
        n_classes, filter_size, n_features_inp, l2, tower_setup)
      self.adjustable_output_assign_data = self._create_adjustable_output_assign_data(tower_setup)
      if self.n_classes_current is None:
        self.n_classes_current = n_classes

      if dilation is None:
        y_pred = conv2d(inp, W) + b
      else:
        y_pred = conv2d_dilated(inp, W, dilation) + b
      self.outputs = [tf.nn.softmax(y_pred, -1, 'softmax')]

      if resize_targets:
        targets = tf.image.resize_nearest_neighbor(targets, tf.shape(y_pred)[1:3])
      if resize_logits:
        y_pred = tf.image.resize_images(y_pred, tf.shape(targets)[1:3])

      pred = tf.argmax(y_pred, axis=3)
      targets = tf.cast(targets, tf.int64)
      targets = tf.squeeze(targets, axis=3)

      # TODO: Void label is not considered in the iou calculation.
      if void_label is not None:
        # avoid nan by replacing void label by 0
        # note: the loss for these cases is multiplied by 0 below
        void_label_mask = tf.equal(targets, void_label)
        no_void_label_mask = tf.logical_not(void_label_mask)
        targets = tf.where(void_label_mask, tf.zeros_like(targets), targets)
      else:
        no_void_label_mask = None

      self.measures = self.create_measures(n_classes, pred, targets)
      self.loss = self.create_loss(loss, fraction, no_void_label_mask, targets, tower_setup, void_label, y_pred)
      self.add_scalar_summary(self.loss, "loss")
