from Forwarding.OneshotForwarder import OneshotForwarder
from datasets.Util.Timer import Timer
from Measures import average_measures
from Log import log
import os
import numpy
from scipy.ndimage.morphology import distance_transform_edt, grey_erosion
import scipy.misc
import numpy as np
import cv2
VOID_LABEL = 255
import pickle
from Forwarding.TeacherAdaptingForwarder import TeacherAdaptingForwarder

class TeacherContAdaptingForwarder(TeacherAdaptingForwarder):
  def __init__(self, engine):
    super(TeacherContAdaptingForwarder, self).__init__(engine)
    self.adapt_flag = 3

