import glob
import time

import tensorflow as tf
from tensorflow.contrib.framework import list_variables

import Constants
import Measures
from Log import log
from Network import Network
from Trainer import Trainer
from Util import load_wider_or_deeper_mxnet_model
from datasets.Forward import forward, online_forward, base_forward, online_forward_cont
from datasets.Loader import load_dataset

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

class Engine(object):
  def __init__(self, config):
    self.config = config
    self.dataset = config.unicode("dataset").lower()
    self.load_init = config.unicode("load_init", "")
    self.load = config.unicode("load", "")
    self.task = config.unicode("task", "train")
    self.use_partialflow = config.bool("use_partialflow", False)
    self.twostream = config.bool("twostream", False)
    self.do_oneshot_or_online_or_offline = self.task in ("teach", "baseline", "teachcont")
    if self.do_oneshot_or_online_or_offline:
      assert config.int("batch_size_eval", 1) == 1
    self.need_train = self.task == "train" or self.do_oneshot_or_online_or_offline or self.task == "forward_train"

    config1 = tf.ConfigProto(allow_soft_placement=True)
    config1.gpu_options.allow_growth = True
    self.session = tf.InteractiveSession(config=config1)
    self.coordinator = tf.train.Coordinator()
    self.valid_data = load_dataset(config, "valid", self.session, self.coordinator)
    if self.need_train:
      self.train_data = load_dataset(config, "train", self.session, self.coordinator)

    self.num_epochs = config.int("num_epochs", 1000)
    self.model = config.unicode("model")
    self.model_base_dir = config.dir("model_dir", "models")
    self.model_dir = self.model_base_dir + self.model + "/"
    self.save = config.bool("save", True)

    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    self.start_epoch = 0
    reuse_variables = None
    if self.need_train:
      freeze_batchnorm = config.bool("freeze_batchnorm", False)
      self.train_network = Network(config, self.train_data, self.global_step, training=True,
                                   use_partialflow=self.use_partialflow,
                                   do_oneshot=self.do_oneshot_or_online_or_offline,
                                   freeze_batchnorm=freeze_batchnorm, name="trainnet")
      reuse_variables = True
    else:
      self.train_network = None
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
      self.test_network = Network(config, self.valid_data, self.global_step, training=False,
                                  do_oneshot=self.do_oneshot_or_online_or_offline, use_partialflow=False,
                                  freeze_batchnorm=True, name="testnet")
    print >> log.v1, "number of parameters:", "{:,}".format(self.test_network.n_params)
    self.trainer = Trainer(config, self.train_network, self.test_network, self.global_step, self.session)
    self.saver = tf.train.Saver(max_to_keep=0, pad_step_number=True)
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()
    tf.train.start_queue_runners(self.session)
    self.load_init_saver = self._create_load_init_saver()
    if not self.do_oneshot_or_online_or_offline:
      self.try_load_weights()
    #put this in again later
    #self.session.graph.finalize()

  def _create_load_init_saver(self):
    if self.load_init != "" and not self.load_init.endswith(".pickle"):
      vars_file = [x[0] for x in list_variables(self.load_init)]
      vars_model = tf.global_variables()
      assert all([x.name.endswith(":0") for x in vars_model])
      vars_intersection = [x for x in vars_model if x.name[:-2] in vars_file]
      vars_missing = [x for x in vars_model if x.name[:-2] not in vars_file]
      if len(vars_missing) > 0:
        print >> log.v1, "the following variables will not be initialized since they are not present in the " \
                         "initialization model", [v.name for v in vars_missing]
      return tf.train.Saver(var_list=vars_intersection)
    else:
      return None

  def try_load_weights(self):
    fn = None
    if self.load != "":
      fn = self.load.replace(".index", "")
    else:
      files = sorted(glob.glob(self.model_dir + self.model + "-*.index"))
      if len(files) > 0:
        fn = files[-1].replace(".index", "")

    if fn is not None:
      print >> log.v1, "loading model from", fn
      self.saver.restore(self.session, fn)
      if self.model == fn.split("/")[-2]:
        self.start_epoch = int(fn.split("-")[-1])
        print >> log.v1, "starting from epoch", self.start_epoch + 1
    elif self.load_init != "":
      if self.load_init.endswith(".pickle"):
        print >> log.v1, "trying to initialize model from wider-or-deeper mxnet model", self.load_init
        load_wider_or_deeper_mxnet_model(self.load_init, self.session)
      elif self.task == 'train' and self.twostream:
        fn = self.load_init
        print >> log.v1, "initializing model from", fn
        assert self.load_init_saver is not None
        self.load_init_saver.restore(self.session, fn)

        variables_2stream = [v for v in tf.all_variables() if '_1' in v.name.split('Adam_1')[0]]
        update_ops = []
        for v in variables_2stream:
            tkns = v.name.split('_1')
            for v2 in tf.all_variables():
                if v2.name == tkns[0]+tkns[1]:
                    update_ops += [v.assign(v2)]
                    break
        self.session.run(update_ops)
      else:
        fn = self.load_init
        print >> log.v1, "initializing model from", fn
        assert self.load_init_saver is not None
        self.load_init_saver.restore(self.session, fn)

  def reset_optimizer(self):
    self.trainer.reset_optimizer()

  @staticmethod
  def run_epoch(step_fn, data, epoch):
    loss_total = 0.0
    n_imgs_per_epoch = data.num_examples_per_epoch()
    measures_accumulated = {}
    n_imgs_processed = 0
    while n_imgs_processed < n_imgs_per_epoch:
      start = time.time()
      loss_summed, measures, n_imgs = step_fn(epoch)
      loss_total += loss_summed

      measures_accumulated = Measures.calc_measures_sum(measures_accumulated, measures)

      n_imgs_processed += n_imgs

      loss_avg = loss_summed / n_imgs
      measures_avg = Measures.calc_measures_avg(measures, n_imgs, data.ignore_classes)
      end = time.time()
      elapsed = end - start

      #TODO: Print proper averages for the measures
      print >> log.v5, n_imgs_processed, '/', n_imgs_per_epoch, loss_avg, measures_avg, "elapsed", elapsed
    loss_total /= n_imgs_processed
    measures_accumulated = Measures.calc_measures_avg(measures_accumulated, n_imgs_processed, data.ignore_classes)
    return loss_total, measures_accumulated

  def train(self):
    assert self.need_train
    print >> log.v1, "starting training"
    for epoch in range(self.start_epoch, self.num_epochs):
      start = time.time()
      train_loss, train_measures = self.run_epoch(self.trainer.train_step, self.train_data, epoch)
      valid_loss, valid_measures = self.run_epoch(self.trainer.validation_step, self.valid_data, epoch)
      end = time.time()
      elapsed = end - start
      train_error_string = Measures.get_error_string(train_measures, "train")
      valid_error_string = Measures.get_error_string(valid_measures, "valid")
      print >> log.v1, "epoch", epoch + 1, "finished. elapsed:", "%.5f" % elapsed, "train_score:", "%.5f" % train_loss,\
          train_error_string, "valid_score:", valid_loss, valid_error_string
      if self.save:
        self.save_model(epoch + 1)

  def eval(self):
    start = time.time()
    valid_loss, measures = self.run_epoch(self.trainer.validation_step, self.valid_data, 0)
    end = time.time()
    elapsed = end - start
    valid_error_string = Measures.get_error_string(measures, "valid")
    print >> log.v1, "eval finished. elapsed:", elapsed, "valid_score:", valid_loss, valid_error_string

  def run(self):
    if self.task == "train":
      self.train()
    elif self.task == "eval":
      self.eval()
    elif self.task in ("forward", "forward_train"):
      if self.task == "forward_train":
        network = self.train_network
        data = self.train_data
      else:
        network = self.test_network
        data = self.valid_data
      save_logits = self.config.bool("save_logits", False)
      save_results = self.config.bool("save_results", True)
      forward(self, network, data, self.dataset, save_results=save_results, save_logits=save_logits)
    elif self.do_oneshot_or_online_or_offline:
      save_logits = self.config.bool("save_logits", False)
      save_results = self.config.bool("save_results", False)
      if self.task == "baseline":
        base_forward(self, save_results=save_results, save_logits=save_logits)
      elif self.task == "teach" :
        online_forward(self, save_results=save_results, save_logits=save_logits)
      elif self.task == "teachcont" :
        online_forward_cont(self, save_results=save_results, save_logits=save_logits)

      else:
        assert False, "Unknown task " + str(self.task)
    else:
      assert False, "Unknown task " + str(self.task)

  def save_model(self, epoch):
    tf.gfile.MakeDirs(self.model_dir)
    self.saver.save(self.session, self.model_dir + self.model, epoch)
