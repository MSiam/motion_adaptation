import os
import pickle
import scipy.io
import time
from collections import deque
from scipy import misc

import numpy as np
import tensorflow as tf

import Util
from datasets.PascalVOC.PascalVOC import PascalVOCDataset, VOID_LABEL
from datasets.Util import Reader

NUM_CLASSES = 2


def postproc_labels_for_objectness(label):
  # 0 is background
  # 255 is void
  # 1-20 are normal classes

  def my_postproc(l):
    l_out = np.zeros_like(l)
    l_out[np.logical_and(l != 0, l != 255)] = 1
    # do we want void here?
    l_out[l == 255] = VOID_LABEL
    return l_out

  label_out, = tf.py_func(my_postproc, [label], [label.dtype])
  label_out.set_shape(label.get_shape())
  return label_out


class PascalVOCInstanceDataset(PascalVOCDataset):
  # Dictionary to keep track of object instances.
  instances = {}
  def __init__(self, config, subset, coord, fraction=1.0):
    super(PascalVOCInstanceDataset, self).__init__(config, subset, coord,
                                                   label_postproc_fn=postproc_labels_for_objectness,
                                                   name="pascalvoc_instance",
                                                   num_classes=NUM_CLASSES, fraction=fraction,
                                                   label_load_fn=self.label_load_fn,
                                                   img_load_fn=self.img_load_fn,
                                                   ignore_classes=[0])

  #Override the default image load since the image paths will be appended by instance numbers
  def img_load_fn(self, img_path):
    path = tf.string_split([img_path], ':').values[0]
    return Reader.load_img_default(path)

  def label_load_fn(self, img_path, label_path):
    def my_create_labels(im_path, label_path):
      return self.create_label(im_path, label_path)

    [label, old_label] = tf.py_func(my_create_labels, [img_path, label_path], [tf.uint8, tf.uint8])
    # labels=Reader.load_label_default(img_path=img_path, label_path=label_path)
    labels = {}
    labels['label'] = label
    labels['old_label'] = old_label
    return labels

  def create_label(self, img_path, label_path):
    segm = self.instances[img_path]
    # Get instance number from the label path.
    inst = img_path.rsplit(":",1)[1]

    old_label = np.zeros((segm['shape'][0], segm['shape'][1], 1))

    [rmin, cmin, rmax, cmax] = segm['bbox']
    old_label[int(rmin):int(rmax), int(cmin):int(cmax), :] = 1

    label = np.zeros_like(old_label)
    mask = segm['mask']
    label[mask[0], mask[1], :] = 1
    if len(np.where(label == 1)[0]) < 20:
      print "Image with a small mask: " + img_path + " instance: " + `inst`

    return label.astype(np.uint8), old_label.astype(np.uint8)

  def get_filename_without_extension(self, img_path):
    file_name = img_path.split("/")[-1]
    file_name_wihout_ext = file_name.split(".")[0]
    return file_name_wihout_ext

  def read_inputfile_lists(self):
    data_list = "train.txt" if self.subset == "train" else "val.txt"
    data_list = "datasets/PascalVOC/" + data_list
    imgs = []
    ans = []
    start = time.time()
    with open(data_list) as f:
      for l in f:
        im, an = l.strip().split()
        im = self.data_dir + im
        an = self.data_dir + an

          # TODO: It takes around 53 seconds to create the instances, so the following method need some optimisations.
        instances, _ = self.get_instances(im)
        # assert len(instances) == len(ans)

        for i in instances:
          img_path = im + ":" + `i`
          imgs.append(img_path)
          ans.append(an)

    end = time.time()
    print "Time taken to read inputs: " + `end - start`
    return imgs, ans

  def get_instances(self, im):
    self.instances[im] = deque()
    instances = []
    anns = []
    instance_segm = None
    file_name_without_ext = self.get_filename_without_extension(im)
    inst_path = self.data_dir + "inst/" + file_name_without_ext + ".mat"

    #Get instances from SBD during training, if they are available
    if self.subset == "train" and os.path.exists(inst_path):
      instance_segm = scipy.io.loadmat(inst_path)['GTinst']['Segmentation'][0][0]
    else:
      file_name_without_ext = self.get_filename_without_extension(im)
      inst_path = self.data_dir + "/SegmentationObject/" + file_name_without_ext + ".png"
      if os.path.exists(inst_path):
        instance_segm = misc.imread(inst_path)
      else:
        print "File: " + im + " does not have any instance annotations."

    if instance_segm is not None:
      inst_labels = np.unique(instance_segm)
      inst_labels = np.setdiff1d(inst_labels,
                                 np.append(np.array(self.ignore_classes), [0, VOID_LABEL]))
      for inst in inst_labels:
        # Create bounding box from segmentation mask.
        rows = np.where(instance_segm == inst)[0]
        cols = np.where(instance_segm == inst)[1]
        rmin = rows.min()
        rmax = rows.max()
        cmin = cols.min()
        cmax = cols.max()
        area = (rmax - rmin) * (cmax - cmin)
        if area > 200:
          anns.append(inst_path)
          instances.append(inst)
          img_path = im + ":" + `inst`
          self.instances[img_path] = {'mask': [rows, cols],
                                 'bbox': [rmin, cmin, rmax, cmax], 'shape': instance_segm.shape}
    return instances, anns

  #Use object detections from Fast-RCNN as the bounding box guidance.
  def get_object_detections(self, im):
    self.instances[im] = deque()
    inst = 0
    instance_segm = None
    anns = []
    file_name_without_ext = self.get_filename_without_extension(im)
    dets_path = self.data_dir + "dets/" + file_name_without_ext + ".pickle"

    if os.path.exists(dets_path):
      boxes = pickle.load(open(dets_path))
      inst_path = self.data_dir + "inst/" + file_name_without_ext + ".mat"
      if os.path.exists(inst_path):
        instance_segm = scipy.io.loadmat(inst_path)['GTinst']['Segmentation'][0][0]
      else:
        file_name_without_ext = self.get_filename_without_extension(im)
        inst_path = self.data_dir + "/SegmentationObject/" + file_name_without_ext + ".png"
        if os.path.exists(inst_path):
          instance_segm = misc.imread(inst_path)
        else:
          print "File: " + im + " does not have any instance annotations."

      if instance_segm is not None:
        for box in boxes:
          if box.shape[0] != 0:
            inst += 1
            ann = inst_path + "_" + `inst`
            anns.append(ann)
            [cmin, rmin, cmax, rmax] = box[0][:4].astype(int)
            bbox_mask = np.zeros_like(instance_segm)

            #The ground truth mask is obtained as the instance with maximum overlap with the bounding box.
            bbox_mask[rmin:rmax, cmin:cmax] = 1
            best_overlap = Util.get_best_overlap(pred_mask=bbox_mask, gt=instance_segm,
                                                 ignore_classes=self.ignore_classes)

            self.instances[ann] = {'mask': np.where(best_overlap != 0), 'bbox': [rmin, cmin, rmax, cmax],
                                   'shape': instance_segm.shape}

            # img = Image.fromarray(bbox_mask)
            # img.save("labels/" + file_name_without_ext + "_bbox.png")
            #
            # img = Image.fromarray(best_overlap)
            # img.save("labels/" + file_name_without_ext + "_mask.png")

    return anns
