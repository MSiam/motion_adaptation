import inspect

import tensorflow as tf

import Measures
import NetworkLayers
import NetworkOutputLayers
from Log import log
from Util_Network import TowerSetup


def get_layer_class(layer_class):
  if hasattr(NetworkLayers, layer_class):
    class_ = getattr(NetworkLayers, layer_class)
  elif hasattr(NetworkOutputLayers, layer_class):
    class_ = getattr(NetworkOutputLayers, layer_class)
  elif hasattr(NetworkRecurrentLayers, layer_class):
    class_ = getattr(NetworkRecurrentLayers, layer_class)
  else:
    assert False, ("Unknown layer class", layer_class)
  return class_


class Network(object):
  def build_tower(self, network_def, x_image, y_ref, void_label, n_classes, tower_setup):
    use_dropout = not tower_setup.is_training
    gpu_str = "/gpu:" + str(tower_setup.gpu)
    if tower_setup.is_main_train_tower:
      print >> log.v4, "inputs:", [x_image.get_shape().as_list()]
    with tf.device(gpu_str), tf.name_scope("tower_gpu_" + str(tower_setup.gpu)):
      output_layer = None
      layers = {}
      for name, layer_def in network_def.items():
        layer_def = layer_def.copy()
        layer_class = layer_def["class"]
        if layer_class == "GraphSection":
          if self.use_partialflow:
            if self.current_graph_section is not None:
              self.current_graph_section.__exit__(None, None, None)
            self.current_graph_section = self.graph_section_manager.new_section()
            self.graph_sections.append(self.current_graph_section)
            self.current_graph_section.__enter__()
          #else:
          #  print >> log.v1, "warning, GraphSection defined, but use_partialflow is False. Ignoring sections"
          continue
        del layer_def["class"]
        class_ = get_layer_class(layer_class)
        spec = inspect.getargspec(class_.__init__)
        args = spec[0]

        if "from" in layer_def:
          inputs = sum([layers[x].outputs for x in layer_def["from"]], [])
          del layer_def["from"]
        else:
          inputs = [x_image]
        if "concat" in layer_def:
          concat = sum([layers[x].outputs for x in layer_def["concat"]], [])
          layer_def["concat"] = concat
        if "alternative_labels" in layer_def:
          layer_def["targets"] = sum([layers[x].out_labels for x in layer_def["alternative_labels"]])
          layer_def["n_classes"] = 2
          del layer_def["alternative_labels"]
        elif class_.output_layer:
          layer_def["targets"] = y_ref
          layer_def["n_classes"] = n_classes
          if "void_label" in args:
            layer_def["void_label"] = void_label
        elif "targets" in args:
          layer_def["targets"] = y_ref
        layer_def["name"] = name
        layer_def["inputs"] = inputs
        if "dropout" in args and not use_dropout:
          layer_def["dropout"] = 0.0
        if "tower_setup" in args:
          layer_def["tower_setup"] = tower_setup

        #check if all args are specified
        defaults = spec[3]
        if defaults is None:
          defaults = []
        n_non_default_args = len(args) - len(defaults)
        non_default_args = args[1:n_non_default_args]  # without self
        for arg in non_default_args:
          assert arg in layer_def, (name, arg)

        layer = class_(**layer_def)

        if tower_setup.is_main_train_tower:
          print >> log.v4, name, "shape:", [l.get_shape().as_list() for l in layer.outputs]
        layers[name] = layer
        if class_.output_layer:
          assert output_layer is None, "Currently only 1 output layer is supported"
          output_layer = layer
      assert output_layer is not None, "No output layer in network"

      n = tf.shape(y_ref)[0]
      assert len(output_layer.outputs) == 1, len(output_layer.outputs)
      loss, measures, y_softmax = output_layer.loss, output_layer.measures, output_layer.outputs[0]
      regularizers_tower = []
      update_ops_tower = []
      for l in layers.values():
        self.summaries += l.summaries
        regularizers_tower += l.regularizers
        update_ops_tower += l.update_ops
      n_params = sum([l.n_params for l in layers.values()])
      return loss, measures, y_softmax, n, n_params, regularizers_tower, update_ops_tower, layers

  def build_network(self, config, x_image, y_ref, void_label, n_classes, is_training, freeze_batchnorm):
    gpus = config.int_list("gpus")
    #only use one gpu for eval
    if not is_training:
      gpus = gpus[:1]
    if self.use_partialflow:
      assert len(gpus) == 1, len(gpus)  # partialflow does not work with multigpu
    network_def = config.dict("network")
    batch_size_tower = self.batch_size / len(gpus)
    assert batch_size_tower * len(gpus) == self.batch_size, (batch_size_tower, len(gpus), self.batch_size)
    loss_summed = measures_accumulated = y_softmax_total = n_total = n_params = None
    tower_losses = []
    tower_regularizers = []
    update_ops = []
    tower_setups = []
    tower_layers = []
    first = True
    for idx, gpu in enumerate(gpus):
      if len(gpus) == 1:
        x_image_tower = x_image
        y_ref_tower = y_ref
        variable_device = "/gpu:0"
      else:
        x_image_tower = x_image[idx * batch_size_tower: (idx + 1) * batch_size_tower]
        y_ref_tower = y_ref[idx * batch_size_tower: (idx + 1) * batch_size_tower]
        variable_device = "/cpu:0"

      is_main_train_tower = is_training and first
      tower_setup = TowerSetup(dtype=config.dtype, gpu=gpu, is_main_train_tower=is_main_train_tower,
                               is_training=is_training, freeze_batchnorm=freeze_batchnorm,
                               variable_device=variable_device, use_update_ops_collection=self.use_partialflow,
                               batch_size=self.batch_size)
      tower_setups.append(tower_setup)

      with tf.variable_scope(tf.get_variable_scope(), reuse=True if not first else None):
        loss, measures, y_softmax, n, n_params_tower, regularizers, update_ops_tower, layers = self.build_tower(
          network_def, x_image_tower, y_ref_tower, void_label, n_classes, tower_setup)

      tower_layers.append(layers)
      tower_losses.append(loss / tf.cast(n, tower_setup.dtype))
      tower_regularizers.append(regularizers)
      if first:
        loss_summed = loss
        measures_accumulated = measures
        y_softmax_total = [y_softmax]
        n_total = n
        update_ops = update_ops_tower
        first = False
        n_params = n_params_tower
      else:
        loss_summed += loss
        Measures.calc_measures_sum(measures_accumulated, measures)
        y_softmax_total.append(y_softmax)
        n_total += n
        update_ops += update_ops_tower
        assert n_params_tower == n_params
    if len(gpus) == 1:
      y_softmax_total = y_softmax_total[0]
    else:
      y_softmax_total = tf.concat(axis=0, values=y_softmax_total, name='y_softmax_total')
    if self.current_graph_section is not None:
      self.current_graph_section.__exit__(None, None, None)
    return tower_losses, tower_regularizers, loss_summed, y_softmax_total, measures_accumulated, n_total, n_params, \
        update_ops, tower_setups, tower_layers

  def __init__(self, config, dataset, global_step, training, do_oneshot, use_partialflow=False, freeze_batchnorm=False,
               name=""):
    with tf.name_scope(name):
      self.use_partialflow = use_partialflow
      if use_partialflow:
        import partialflow
        self.graph_section_manager = partialflow.GraphSectionManager(verbose=False)
        self.graph_sections = []
      else:
        self.graph_section_manager = None
      self.current_graph_section = None
      if training:
        self.batch_size = config.int("batch_size")
        self.chunk_size = config.int("chunk_size", -1)
      else:
        assert freeze_batchnorm
        self.chunk_size = config.int("eval_chunk_size", -1)
        if self.chunk_size == -1:
          self.chunk_size = config.int("chunk_size", -1)

        do_multi_sample_testing = config.int("n_test_samples", -1) != -1
        if do_multi_sample_testing:
          self.batch_size = 1
        else:
          self.batch_size = config.int("batch_size_eval", -1)
          if self.batch_size == -1:
            self.batch_size = config.int("batch_size")
      n_classes = dataset.num_classes()
      if config.bool("adjustable_output_layer", False):
        n_classes = None
      self.global_step = global_step
      inputs_tensors_dict = dataset.create_input_tensors_dict(self.batch_size)
      #inputs and labels are not optional
      inputs = inputs_tensors_dict["inputs"]
      labels = inputs_tensors_dict["labels"]
      self.raw_labels = inputs_tensors_dict.get("raw_labels", None)
      self.index_imgs = inputs_tensors_dict.get("index_imgs", None)
      self.tags = inputs_tensors_dict.get("tags")

      void_label = dataset.void_label()
      #important: first inputs_and_labels (which creates summaries) and then access summaries
      self.summaries = []
      self.summaries += dataset.summaries
      self.losses, self.regularizers, self.loss_summed, self.y_softmax, self.measures_accumulated, self.n_imgs, \
          self.n_params, self.update_ops, self.tower_setups, self.tower_layers = self.build_network(
            config, inputs, labels, void_label, n_classes, training, freeze_batchnorm)

  def get_output_layer(self):
    layers = self.tower_layers[0]
    output_layers = [l for l in layers.values() if l.output_layer]
    assert len(output_layers) == 1
    return output_layers[0]
