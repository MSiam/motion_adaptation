#!/usr/bin/env python

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax
from scipy.ndimage import imread
from scipy.misc import imsave
import cPickle
import numpy
import glob
import os
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import sys
import pdb
imgs_path = "/home/eren/Data/FBMS/Testset/JPEGImages/"
annots_path = "/home/eren/Data/FBMS/Testset/Annotations/"
preds_path_prefix = "/home/eren/Work/motion_adaptation/forwarded/"


def convert_path(inp):
  sp = inp.split("/")
  fwd_idx = sp.index("forwarded")

  seq = sp[fwd_idx + 3]
  fn = sp[-1]
  im_path = imgs_path + seq + "/" + fn.replace(".pickle", ".jpg")
  gt_path = annots_path + seq + "/" + fn.replace(".pickle", ".png")

  sp[fwd_idx + 1] += "_crf"
  sp[-1] = sp[-1].replace(".pickle", ".png")
  out_path = "/".join(sp)
  return im_path, gt_path, out_path


def mkdir_p(d):
  try:
    os.makedirs(d)
  except OSError as err:
    if err.errno != 17:
      raise


def apply_crf(im, pred):
  im = numpy.ascontiguousarray(im)
  pred = numpy.ascontiguousarray(pred.swapaxes(0, 2).swapaxes(1, 2))

  d = dcrf.DenseCRF2D(im.shape[1], im.shape[0], 2)  # width, height, nlabels
  unaries = unary_from_softmax(pred, scale=1.0)
  d.setUnaryEnergy(unaries)

  #print im.shape
  # print annot.shape
  #print pred.shape

  d.addPairwiseGaussian(sxy=0.220880737269, compat=1.24845093352)
  d.addPairwiseBilateral(sxy=22.3761305044, srgb=7.70254062277, rgbim=im, compat=1.40326787165)
  processed = d.inference(12)
  res = numpy.argmax(processed, axis=0).reshape(im.shape[0], im.shape[1])

  return res


def do_seq(seq, model, save=True):
  preds_path = preds_path_prefix + model + "/valid/"
  files = sorted(glob.glob(preds_path + seq + "/*.pickle"))
  ious = []
  fmes= []
  recalls= []
  precs= []
  for f in files:
    pred_path = f
    im_path, gt_path, out_path = convert_path(f)
    pred = cPickle.load(open(pred_path))
    im = imread(im_path)
    res = apply_crf(im, pred).astype("uint8") * 255
    # before = numpy.argmax(pred, axis=2)
    if save:
      dir_ = "/".join(out_path.split("/")[:-1])
      mkdir_p(dir_)
      imsave(out_path, res)

    #compute iou as well
    if not os.path.exists(gt_path):
        gt_path = gt_path.split('.')[0]+'_gt.png'
    groundtruth = imread(gt_path)
    I = numpy.logical_and(res == 255, groundtruth == 255).sum()
    U = numpy.logical_or(res == 255, groundtruth == 255).sum()

    if U==0:
        recall = 1.0
        precision = 1.0

    IOU = float(I) / U
    ious.append(IOU)

    T = (groundtruth==255).sum()
    P = (res==255).sum()

    if T == 0:
      recall = 1.0
    else:
      recall = float(I) / T

    if P == 0:
      precision = 1.0
    else:
      precision = float(I) / P
    if recall+precision<0.05:
        fmeasure=0
    else:
        fmeasure= 2*recall*precision/ (recall+precision)
    fmes.append(fmeasure)
    recalls.append(recall)
    precs.append(precision)

    print out_path, "IOU", IOU

    # plt.imshow(before)
    # plt.figure()
    # plt.imshow(res)
    # plt.show()
#  return numpy.mean(ious[1:-1]), numpy.mean(fmes[1:-1]), numpy.mean(precs[1:-1]), numpy.mean(recalls[1:-1])
  return numpy.mean(ious), numpy.mean(fmes), numpy.mean(precs), numpy.mean(recalls)


def main():

  save = True
  assert len(sys.argv) == 2
  model = sys.argv[1]
  seqs= os.listdir('/home/eren/Data/FBMS/Testset/JPEGImages/')

  ious = []
  fmes= []
  precs= []
  recs= []
  for seq in seqs:
    iou, fms, prec, rec = do_seq(seq, model, save=save)
    print iou, fms, prec, rec
    ious.append(iou)
    fmes.append(fms)
    precs.append(prec)
    recs.append(rec)

#  ious, fmes, precs, recs = Parallel(n_jobs=20)(delayed(do_seq)(seq, model, save=save) for seq in seqs)

  print seqs
  print ious, fmes, precs, recs
  print numpy.mean(ious), numpy.mean(fmes), numpy.mean(precs), numpy.mean(recs)


if __name__ == "__main__":
  main()
