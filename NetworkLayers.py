import numpy
import tensorflow as tf
from tensorflow.python.training import moving_averages
from datasets.Util.Util import smart_shape

from Util_Network import conv2d, max_pool, global_avg_pool, apply_dropout, prepare_input, \
  prepare_collapsed_input_and_dropout, get_activation, create_batch_norm_vars, create_bilinear_upsampling_weights, \
  conv2d_dilated

BATCH_NORM_DECAY_DEFAULT = 0.95
BATCH_NORM_EPSILON = 1e-5
L2_DEFAULT = 1e-4


class Layer(object):
  output_layer = False

  def __init__(self):
    self.summaries = []
    self.regularizers = []
    self.update_ops = []
    self.n_params = 0

  def add_scalar_summary(self, op, name):
    summary = tf.summary.scalar(name, op)
    self.summaries.append(summary)

  def create_and_apply_batch_norm(self, inp, n_features, decay, tower_setup, scope_name="bn"):
    beta, gamma, moving_mean, moving_var = create_batch_norm_vars(n_features, tower_setup, scope_name)
    self.n_params += 2 * n_features
    if tower_setup.is_main_train_tower:
      assert tower_setup.is_training
    if tower_setup.is_training and not tower_setup.freeze_batchnorm:
      xn, batch_mean, batch_var = tf.nn.fused_batch_norm(inp, gamma, beta, epsilon=BATCH_NORM_EPSILON, is_training=True)
      if tower_setup.is_main_train_tower:
        update_op1 = moving_averages.assign_moving_average(
          moving_mean, batch_mean, decay, zero_debias=False, name='mean_ema_op')
        update_op2 = moving_averages.assign_moving_average(
          moving_var, batch_var, decay, zero_debias=False, name='var_ema_op')
        if tower_setup.use_update_ops_collection:
          tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_op1)
          tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_op2)
        else:
          self.update_ops.append(update_op1)
          self.update_ops.append(update_op2)
      return xn
    else:
      xn = tf.nn.batch_normalization(inp, moving_mean, moving_var, beta, gamma, BATCH_NORM_EPSILON)
      return xn

  def create_weight_variable(self, name, shape, l2, tower_setup):
    with tf.device(tower_setup.variable_device):
      # He initialization
      initializer = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False)
      self.n_params += numpy.prod(shape)
      W = tf.get_variable(name, shape, tower_setup.dtype, initializer)
      if l2 > 0.0:
        self.regularizers.append(l2 * tf.nn.l2_loss(W))
      # self.add_scalar_summary(tf.reduce_max(tf.abs(W)), "W_abs_max")
      return W

  # adapted from https://github.com/MarvinTeichmann/tensorflow-fcn/blob/master/fcn16_vgg.py
  def create_transposed_conv_weight_variable(self, name, shape, l2, tower_setup):
    with tf.device(tower_setup.variable_device):
      weights = create_bilinear_upsampling_weights(shape)
      initializer = tf.constant_initializer(value=weights, dtype=tf.float32)
      self.n_params += numpy.prod(shape)
      W = tf.get_variable(name, shape, tower_setup.dtype, initializer)
      if l2 > 0.0:
        self.regularizers.append(l2 * tf.nn.l2_loss(W))
      return W

  def create_bias_variable(self, name, shape, tower_setup):
    with tf.device(tower_setup.variable_device):
      initializer = tf.constant_initializer(0.0, dtype=tower_setup.dtype)
      self.n_params += numpy.prod(shape)
      return tf.get_variable(name, shape, tower_setup.dtype, initializer)


class Conv(Layer):
  def __init__(self, name, inputs, n_features, tower_setup, old_order=False, filter_size=(3, 3),
               strides=(1, 1), dilation=None, pool_size=(1, 1), pool_strides=None, activation="relu", dropout=0.0,
               batch_norm=False, bias=False, batch_norm_decay=BATCH_NORM_DECAY_DEFAULT, l2=L2_DEFAULT):
    super(Conv, self).__init__()
    # mind the order of dropout, conv, activation and batchnorm!
    # default: batchnorm -> activation -> dropout -> conv -> pool
    # if old_order: dropout -> conv -> batchnorm -> activation -> pool

    curr, n_features_inp = prepare_input(inputs)

    filter_size = list(filter_size)
    strides = list(strides)
    pool_size = list(pool_size)
    if pool_strides is None:
      pool_strides = pool_size

    with tf.variable_scope(name):
      W = self.create_weight_variable("W", filter_size + [n_features_inp, n_features], l2, tower_setup)
      b = None
      if bias:
        b = self.create_bias_variable("b", [n_features], tower_setup)

      if old_order:
        curr = apply_dropout(curr, dropout)
        if dilation is None:
          curr = conv2d(curr, W, strides)
        else:
          curr = conv2d_dilated(curr, W, dilation)
        if bias:
          curr += b
        if batch_norm:
          curr = self.create_and_apply_batch_norm(curr, n_features, batch_norm_decay, tower_setup)
        curr = get_activation(activation)(curr)
      else:
        if batch_norm:
          curr = self.create_and_apply_batch_norm(curr, n_features_inp, batch_norm_decay, tower_setup)
        curr = get_activation(activation)(curr)
        curr = apply_dropout(curr, dropout)
        if dilation is None:
          curr = conv2d(curr, W, strides)
        else:
          curr = conv2d_dilated(curr, W, dilation)
        if bias:
          curr += b

      if pool_size != [1, 1]:
        curr = max_pool(curr, pool_size, pool_strides)
    self.outputs = [curr]


class ResidualUnit2(Layer):
  def __init__(self, name, inputs, tower_setup, n_convs=2, n_features=None, dilations=None, strides=None,
               filter_size=None, activation="relu", batch_norm_decay=BATCH_NORM_DECAY_DEFAULT, l2=L2_DEFAULT):
    super(ResidualUnit2, self).__init__()
    # TODO: dropout
    curr, n_features_inp = prepare_input(inputs)
    res = curr
    assert n_convs >= 1, n_convs

    if dilations is not None:
      assert strides is None
    elif strides is None:
      strides = [[1, 1]] * n_convs
    if filter_size is None:
      filter_size = [[3, 3]] * n_convs
    if n_features is None:
      n_features = n_features_inp
    if not isinstance(n_features, list):
      n_features = [n_features] * n_convs

    with tf.variable_scope(name):
      curr = self.create_and_apply_batch_norm(curr, n_features_inp, batch_norm_decay, tower_setup, "bn0")
      curr = get_activation(activation)(curr)

      if strides is None:
        strides_res = [1, 1]
      else:
        strides_res = numpy.prod(strides, axis=0).tolist()
      if (n_features[-1] != n_features_inp) or (strides_res != [1, 1]):
        W0 = self.create_weight_variable("W0", [1, 1] + [n_features_inp, n_features[-1]], l2, tower_setup)
        if dilations is None:
          res = conv2d(curr, W0, strides_res)
        else:
          res = conv2d(curr, W0)

      W1 = self.create_weight_variable("W1", filter_size[0] + [n_features_inp, n_features[0]], l2, tower_setup)
      if dilations is None:
        curr = conv2d(curr, W1, strides[0])
      else:
        curr = conv2d_dilated(curr, W1, dilations[0])
      for idx in xrange(1, n_convs):
        curr = self.create_and_apply_batch_norm(curr, n_features[idx - 1], batch_norm_decay,
                                                tower_setup, "bn" + str(idx + 1))
        curr = get_activation(activation)(curr)
        Wi = self.create_weight_variable("W" + str(idx + 1), filter_size[idx] + [n_features[idx - 1], n_features[idx]],
                                         l2, tower_setup)
        if dilations is None:
          curr = conv2d(curr, Wi, strides[idx])
        else:
          curr = conv2d_dilated(curr, Wi, dilations[idx])

    curr += res
    self.outputs = [curr]


class Upsampling(Layer):
  def __init__(self, name, inputs, tower_setup, n_features, concat, activation="relu",
               filter_size=(3, 3), batch_norm_decay=BATCH_NORM_DECAY_DEFAULT, l2=L2_DEFAULT):
    super(Upsampling, self).__init__()
    filter_size = list(filter_size)
    assert isinstance(concat, list)
    assert len(concat) > 0
    curr, n_features_inp = prepare_input(inputs)
    concat_inp, n_features_concat = prepare_input(concat)

    curr = tf.image.resize_nearest_neighbor(curr, tf.shape(concat_inp)[1:3])
    curr = tf.concat([curr, concat_inp], axis=3)
    n_features_curr = n_features_inp + n_features_concat

    with tf.variable_scope(name):
      W = self.create_weight_variable("W", filter_size + [n_features_curr, n_features], l2, tower_setup)
      b = self.create_bias_variable("b", [n_features], tower_setup)
      curr = conv2d(curr, W) + b
      curr = get_activation(activation)(curr)

    self.outputs = [curr]


class FullyConnected(Layer):
  def __init__(self, name, inputs, n_features, tower_setup, activation="relu", dropout=0.0, batch_norm=False,
               batch_norm_decay=BATCH_NORM_DECAY_DEFAULT, l2=L2_DEFAULT):
    super(FullyConnected, self).__init__()
    inp, n_features_inp = prepare_collapsed_input_and_dropout(inputs, dropout)
    with tf.variable_scope(name):
      if batch_norm:
        inp = tf.expand_dims(inp, axis=0)
        inp = tf.expand_dims(inp, axis=0)
        inp = self.create_and_apply_batch_norm(inp, n_features_inp, batch_norm_decay, tower_setup)
        inp = tf.squeeze(inp, axis=[0, 1])
      W = self.create_weight_variable("W", [n_features_inp, n_features], l2, tower_setup)
      b = self.create_bias_variable("b", [n_features], tower_setup)
      z = tf.matmul(inp, W) + b
      h = get_activation(activation)(z)
    self.outputs = [h]


class Collapse(Layer):
  def __init__(self, name, inputs, tower_setup, activation="relu", batch_norm_decay=BATCH_NORM_DECAY_DEFAULT):
    super(Collapse, self).__init__()
    curr, n_features_inp = prepare_input(inputs)
    with tf.variable_scope(name):
      inp = self.create_and_apply_batch_norm(curr, n_features_inp, batch_norm_decay, tower_setup)
      h_act = get_activation(activation)(inp)
      out = global_avg_pool(h_act)
    self.outputs = [out]


class SiameseConcat(Layer):
  def __init__(self, name, inputs, tower_setup):
    super(SiameseConcat, self).__init__()
    curr, n_features_inp = prepare_input(inputs)
    # old_shape = smart_shape(curr)
    # batch_size = old_shape[0]
    out = tf.reshape(curr, [-1, n_features_inp * 2])
    self.outputs = [out]

class DoNothingLayer(Layer):
  def __init__(self, name, inputs, tower_setup):
    super(DoNothingLayer, self).__init__()
    curr, n_features_inp = prepare_input(inputs)
    # old_shape = smart_shape(curr)
    # batch_size = old_shape[0]
    # out = tf.reshape(curr, [-1, n_features_inp * 2])
    self.outputs = [curr]

class ExpandedSiameseConcat(Layer):
  def __init__(self, name, inputs, targets, tower_setup):
    super(ExpandedSiameseConcat, self).__init__()
    curr, n_features_inp = prepare_input(inputs)
    size = smart_shape(curr)[0]

    def Expand(idx):
      anchor = curr[idx, :]
      anchor_class = targets[idx]
      classes,_ = tf.unique(targets)
      class_division = tf.cast(tf.equal(targets, anchor_class), tf.int32)
      partitioned_output = tf.dynamic_partition(curr, class_division, 2)
      partitioned_targets = tf.dynamic_partition(targets, class_division, 2)

      # Positives
      positives = partitioned_output[1]
      size_positives = smart_shape(positives)[0]
      anchor_positive_repmat = tf.reshape(tf.tile(anchor,[size_positives]),[size_positives,-1])
      positives_combined = tf.concat((anchor_positive_repmat,positives),1)
      new_targets_positive = tf.ones([smart_shape(positives_combined)[0]],dtype=tf.int32)

      # Negatives
      negative_size = smart_shape(classes)[0]

      def Get_negatives(neg_idx):
        curr_neg_class = classes[neg_idx]
        neg_class_division = tf.cast(tf.equal(targets, curr_neg_class), tf.int32)
        neg_partitioned_output = tf.dynamic_partition(curr, neg_class_division, 2)
        negative_set = neg_partitioned_output[1]
        size_negative_set = smart_shape(negative_set)[0]
        random_negative_idx = tf.random_shuffle(tf.range(1, size_negative_set))[0]
        random_negative = negative_set[random_negative_idx,:]
        return random_negative

      looper = tf.range(0, anchor_class)
      iter_val = tf.minimum(anchor_class+1,negative_size)
      looper = tf.concat([looper,tf.range(iter_val,negative_size)],0)
      negatives = tf.map_fn(Get_negatives, looper, dtype=tf.float32)
      size_negatives = smart_shape(negatives)[0]
      anchor_negative_repmat = tf.reshape(tf.tile(anchor, [size_negatives]), [size_negatives, -1])
      negatives_combined = tf.concat((anchor_negative_repmat,negatives),1)
      new_targets_negative = tf.zeros([smart_shape(negatives_combined)[0]],dtype=tf.int32)

      all_combined = tf.concat((positives_combined,negatives_combined),0)
      new_targets_combined = tf.concat((new_targets_positive,new_targets_negative),0)
      return all_combined, new_targets_combined

    expanded,new_targets = tf.map_fn(Expand, tf.range(0, size), dtype=(tf.float32,tf.int32))
    expanded = tf.reshape(expanded, [-1, n_features_inp * 2])
    new_targets = tf.reshape(new_targets, [-1])

    # new_shape = smart_shape(expanded)
    # tower_setup.is_training*860 + 100
    new_shape = [tower_setup.is_training*896 + 64,1000]
    new_targets.set_shape([new_shape[0]])
    expanded.set_shape(new_shape)
    # self.outputs = [expanded]
    # self.out_labels = new_targets

    def if_training():
      return new_targets, expanded

    def if_not_training():
      ahah = tf.concat([curr,curr],1)
      ahah.set_shape([64,1000])
      return targets, ahah

    self.out_labels, rar = tf.cond(tf.cast(tower_setup.is_training,tf.bool), if_training, if_not_training)
    self.outputs = [rar]
    # self.n_classes = 2
    # inputs = [expanded]
    # targets = new_targets
    # n_classes = 2

    # # Print - debug example code
    # def test(a):
    #   print(a)
    #   return numpy.array([5], dtype="int32")
    #
    # t, = tf.py_func(test, [self.out_labels], [tf.int32])
    # with tf.control_dependencies([t]):
    #   expanded = tf.identity(expanded)
    #
    # self.outputs = [expanded]