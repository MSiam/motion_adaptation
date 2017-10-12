import tensorflow as tf

import Constants
from Log import log
from Util import average_gradients

PROFILE = False
if PROFILE:
  first_run = True


def get_options():
  global first_run
  if PROFILE and not first_run:
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    return run_options, run_metadata
  else:
    return None, None


class Trainer(object):
  def __init__(self, config, train_network, test_network, global_step, session):
    self.measures = config.unicode_list("measures", [])
    self.opt_str = config.unicode("optimizer", "adam").lower()
    self.train_network = train_network
    self.test_network = test_network
    self.session = session
    self.global_step = global_step
    self.learning_rates = config.int_key_dict("learning_rates")
    assert 1 in self.learning_rates, "no initial learning rate specified"
    self.curr_learning_rate = self.learning_rates[1]
    self.lr_var = tf.placeholder(config.dtype, shape=[], name="learning_rate")
    self.loss_scale_var = tf.placeholder_with_default(1.0, shape=[], name="loss_scale")
    self.opt, self.reset_opt_op = self.create_optimizer(config)
    if train_network is not None:
      if train_network.use_partialflow:
        self.prepare_partialflow()
        self.step_op = tf.no_op("step")
      else:
        self.step_op = self.create_step_op()
      if len(self.train_network.update_ops) == 0:
        self.update_ops = []
      else:
        self.update_ops = self.train_network.update_ops
    else:
      self.step_op = None
      self.update_ops = None
    self.summary_writer, self.summary_op, self.summary_op_test = self.init_summaries(config)

  def create_optimizer(self, config):
    momentum = config.float("momentum", 0.9)
    if self.opt_str == "sgd_nesterov":
      return tf.train.MomentumOptimizer(self.lr_var, momentum, use_nesterov=True), None
    elif self.opt_str == "sgd_momentum":
      return tf.train.MomentumOptimizer(self.lr_var, momentum), None
    elif self.opt_str == "sgd":
      return tf.train.GradientDescentOptimizer(self.lr_var), None
    elif self.opt_str == "adam":
      opt = tf.train.AdamOptimizer(self.lr_var)
      all_vars = tf.global_variables()
      opt_vars = [v for v in all_vars if "Adam" in v.name]
      reset_opt_op = tf.variables_initializer(opt_vars, "reset_optimizer")
      return opt, reset_opt_op
    else:
      assert False, ("unknown optimizer", self.opt_str)

  def reset_optimizer(self):
    assert self.opt_str == "adam", "reset not implemented for other optimizers yet"
    assert self.reset_opt_op is not None
    self.session.run(self.reset_opt_op)

  def prepare_partialflow(self):
    sm = self.train_network.graph_section_manager
    losses = self.train_network.losses
    regularizers = self.train_network.regularizers
    assert len(losses) == 1
    assert len(regularizers) == 1
    loss = losses[0] + tf.add_n(regularizers[0])
    loss *= self.loss_scale_var
    sm.add_training_ops(self.opt, loss, verbose=False, global_step=self.global_step)
    sm.prepare_training()
    #for sec in self.train_network.graph_sections:
    #  print sec.get_tensors_to_feed()
    #for sec in self.train_network.graph_sections:
    #  print sec.get_tensors_to_cache()

  def create_step_op(self):
    losses, regularizers, setups = self.train_network.losses, self.train_network.regularizers, \
                                   self.train_network.tower_setups
    assert len(losses) == len(regularizers)
    assert all(len(regularizers[0]) == len(x) for x in regularizers)
    if len(regularizers[0]) > 0:
      regularizers = [tf.add_n(x) for x in regularizers]
    losses_with_regularizers = [l + r for l, r in zip(losses, regularizers)]
    losses_with_regularizers = [x * self.loss_scale_var for x in losses_with_regularizers]
    tower_grads = []
    for l, s in zip(losses_with_regularizers, setups):
      gpu_str = "/gpu:" + str(s.gpu)
      with tf.device(gpu_str), tf.name_scope("tower_gpu_" + str(s.gpu) + "_opt"):
        tower_grads.append(self.opt.compute_gradients(l))
    if len(losses) == 1:
      grads = tower_grads[0]
    else:
      # average the gradients over the towers
      grads = average_gradients(tower_grads)
    step_op = self.opt.apply_gradients(grads, global_step=self.global_step)
    return step_op

  def init_summaries(self, config):
    summdir = config.dir("summary_dir", "summaries")
    model = config.unicode("model")
    summdir += model + "/"
    tf.gfile.MakeDirs(summdir)
    summary_writer = tf.summary.FileWriter(summdir, self.session.graph)
    if config.bool("write_summaries", True) and self.train_network is not None \
       and len(self.train_network.summaries) > 0:
      # better do not merge ALL summaries, since otherwise we get summaries from different networks
      # and might execute (parts of) the test network while training
      # self.summary_op = tf.merge_all_summaries()
      # atm we only collect summaries from the train network
      summary_op = tf.summary.merge(self.train_network.summaries)
      summary_op_test = tf.summary.merge(self.test_network.summaries)
    else:
      summary_op = None
      summary_op_test = None
    return summary_writer, summary_op, summary_op_test

  # for profiling
  def handle_run_metadata(self, metadata):
    if not PROFILE:
      return
    global first_run
    if first_run:
      first_run = False
    else:
      self.summary_writer.add_run_metadata(metadata, 'profile', 0)
      self.summary_writer.flush()
      from tensorflow.python.client import timeline
      tl = timeline.Timeline(metadata.step_stats)
      ctf = tl.generate_chrome_trace_format()
      with open('timeline.json', 'w') as f:
        f.write(ctf)
      quit()

  def validation_step(self, _):
    ops = [self.test_network.loss_summed, self.test_network.measures_accumulated, self.test_network.n_imgs]
    if 'clicks' in self.measures:
      ops.append(self.test_network.tags)

    if self.summary_op_test is not None:
      ops.append(self.summary_op_test)

    res = self.session.run(ops)
    if self.summary_op_test is not None:
      summary_str = res[-1]
      res = res[:-1]
      self.summary_writer.add_summary(summary_str, global_step=None)

    if len(res) > 3:
      loss_summed, measures_accumulated, n_imgs, tags = res
      measures_accumulated[Constants.CLICKS] = tags
    else:
      loss_summed, measures_accumulated, n_imgs = res

    return loss_summed, measures_accumulated, n_imgs

  def adjust_learning_rate(self, epoch, learning_rate=None):
    if learning_rate is None:
      key = max([k for k in self.learning_rates.keys() if k <= epoch + 1])
      new_lr = self.learning_rates[key]
    else:
      new_lr = learning_rate
    if self.curr_learning_rate != new_lr:
      print >> log.v1, "changing learning rate to", new_lr
      self.curr_learning_rate = new_lr

  def train_step(self, epoch, feed_dict=None, loss_scale=1.0, learning_rate=None):
    self.adjust_learning_rate(epoch, learning_rate)
    if feed_dict is None:
      feed_dict = {}
    else:
      feed_dict = feed_dict.copy()
    feed_dict[self.lr_var] = self.curr_learning_rate
    feed_dict[self.loss_scale_var] = loss_scale

    ops = self.update_ops + [self.global_step, self.step_op, self.train_network.loss_summed,
                             self.train_network.measures_accumulated, self.train_network.n_imgs]

    if Constants.CLICKS in self.measures:
      ops.append(self.train_network.tags)

    if self.summary_op is not None:
      ops.append(self.summary_op)

    if self.train_network.use_partialflow:
      res = self.train_network.graph_section_manager.run_full_cycle(
        self.session, fetches=ops, basic_feed=feed_dict)
    else:
      run_options, run_metadata = get_options()
      res = self.session.run(ops, feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)
      self.handle_run_metadata(run_metadata)

    #remove update outputs
    res = res[len(self.update_ops):]

    if self.summary_op is not None:
      summary_str = res[-1]
      res = res[:-1]
      step = res[0]
      self.summary_writer.add_summary(summary_str, step)

    if len(res) > 5:
      _, _, loss_summed, measures_accumulated, n_imgs, tags = res
      measures_accumulated[Constants.CLICKS] = tags
    else:
      _, _, loss_summed, measures_accumulated, n_imgs = res

    return loss_summed, measures_accumulated, n_imgs
