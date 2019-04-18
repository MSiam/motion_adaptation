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
from PIL import Image
from Tkinter import *
import threading
import PIL.Image, PIL.ImageTk

class TeacherAdaptingForwarder(OneshotForwarder):
  def __init__(self, engine):
    super(TeacherAdaptingForwarder, self).__init__(engine)

    self.n_adaptation_steps = self.config.int("n_adaptation_steps", 12)
    self.adaptation_learning_rate = self.config.float("adaptation_learning_rate")
    self.adaptation_loss_scale = self.config.float("adaptation_loss_scale", 0.1)
    self.mot_dir= self.config.unicode("targets_path", "")
    self.few_shot_samples = self.config.int("few_shot_samples", 1)
    self.dataset = self.config.unicode("davis_data_dir", "")

    self.adapt_flag = 1
    self.seqs = sorted(os.listdir(self.mot_dir))

    self.cap = cv2.VideoCapture(int(self.config.cam_idx))
    self.frame = None
    self.overlay = None
    self.flo_w = 512; self.flo_h = 384
    self.panel_img = None

    self.root = Tk()
    labelfont = ('times', 15, 'bold')
    lbl = Label(self.root, text='Classes Learned')
    lbl.config(font=labelfont)
    lbl.grid(row=0, column=1)

    btns = self.create_buttons()
    self.current_sq = -1

    self.stopEvent = threading.Event()
    thread = threading.Thread(target=self.videoLoop, args=())
    thread.start()

    self.root.wm_title("MotAdapt Inference Phase")
    self.root.wm_protocol("WM_DELETE_WINDOW", self.onClose)

    self.segmentFlag = False
    self.adaptFirst = False

  def onClick(self, idx):
    self.current_sq = idx - 1
    self.adaptFirst = True

  def create_buttons(self):
    labelfont = ('times', 15, 'bold')
    idx = 1
    btns = []
    for sq in sorted(os.listdir(self.mot_dir)):
        btns.append(Button(self.root, text=sq.split('sq_')[1],
                           command=lambda inst = self, idx = idx: inst.onClick(idx),
                           width=20))

        btns[-1].config(font=labelfont)
        btns[-1].grid(row=idx, column=1)
        idx += 1

    self.nrows = idx

  def set_panel(self, panel, photo, r, c):
    if panel is None:
        panel = Label(self.root, image=photo)
        panel.image = photo
        panel.grid(row=r, column=c, rowspan=self.nrows)
    else:
        panel.configure(image=photo)
        panel.image = photo
    return panel

  def videoLoop(self):
    while not self.stopEvent.is_set():
        _, self.frame = self.cap.read()
        if self.frame is not None:
            self.frame = cv2.resize(self.frame,
                                    (self.flo_w, self.flo_h))
            if self.segmentFlag:
                self.forward_(self.adaptFirst)
                photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.overlay[:,:,::-1]))
            else:
                photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.frame[:,:,::-1]))

            if self.adaptFirst:
                self.forward_(self.adaptFirst)
                self.adaptFirst = False
                self.segmentFlag = True

            self.panel_img = self.set_panel(self.panel_img, photo, 0, 0)

  def onClose(self):
    self.segmentFlag = False
    self.stopEvent.set()
    self.root.quit()

  def PIL2array(self, img):
    return np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0], 4)

  def create_overlay(self, img, mask, colors):
    im= PIL.Image.fromarray(np.uint8(img))
    im= im.convert('RGBA')

    mask_color= np.zeros((mask.shape[0], mask.shape[1],3))
    mask_color[mask==colors[1],2]=255

    overlay= PIL.Image.fromarray(np.uint8(mask_color))
    overlay= overlay.convert('RGBA')

    im= PIL.Image.blend(im, overlay, 0.7)
    blended_arr= self.PIL2array(im)[:,:,:3]
    img2= img.copy()
    img2[mask!=colors[0],:]= blended_arr[mask!=colors[0],:]
    return img2

  def visualize(self, frame, ys_argmax_val):
    overlay = self.create_overlay(frame, ys_argmax_val[0, :, :, 0], [0, 1])
    return overlay

  def _oneshot_forward_video(self, video_idx, save_logits):
    with Timer():
      # Test Network Variables + Resize output to same shape of Labels
      self.network = self.engine.test_network
      self.targets = self.network.raw_labels
      self.ys = self.network.y_softmax
      self.ys = self._adjust_results_to_targets(self.ys, self.targets)
      self.data = self.val_data

      self.root.mainloop()

  def forward_(self, first):
      # Probability Map Function
      def get_posteriors():
          n_, _, _, logits_val_, _ = self._process_forward_minibatch(
            data, network, save_logits=False, save_results=False, targets=targets, ys=ys, start_frame_idx=t)
          assert n_ == 1
          return logits_val_[0]

      if first:
          files_annotations = sorted(os.listdir(self.mot_dir + \
                                        self.seqs[self.current_sq] + '/'))

          for t in range(self.few_shot_samples):
              f = self.mot_dir + self.seqs[self.current_sq] + \
                      '/' + files_annotations[t]

              mask = cv2.imread(f, 0)
              last_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
              last_mask[mask==255] = 1
              last_mask[mask==128] = VOID_LABEL
              last_mask = np.expand_dims(last_mask, axis=2)
              # ToDo: Modify in data video_idx to new one
              loss = self._adapt(0, t, last_mask, get_posteriors)
              print('Adapting on frame ', t, 'with loss = ', loss)

          print("Finished Adaptation")
      else:
          self.data.camera = True
          self.data.current_frame = self.frame[:, :, ::-1]
          _, _, ys_argmax_val, posteriors_val, _, _ = self._process_forward_minibatch(
              self.data, self.network, False, False, self.targets, self.ys, start_frame_idx=0)

          self.overlay = self.visualize(self.frame, ys_argmax_val)

  def _adapt(self, video_idx, frame_idx, last_mask, get_posteriors_fn):
    """
    adapt_flag (int): 0:do not adapt, 1:adapt with hard labels based on teacher,
                      2:adapt on hard labels from last mask, 3:use continuous labels
    """
    adaptation_target = last_mask
    self.train_data.set_video_idx(video_idx)

    # Start Adapting based on number of adaptation_steps
    for idx in xrange(self.n_adaptation_steps):
      feed_dict = self.train_data.feed_dict_for_video_frame(frame_idx, with_annotations=True)
      feed_dict[self.train_data.get_label_placeholder()] = adaptation_target
      loss_scale = self.adaptation_loss_scale
      adaption_frame_idx = frame_idx
      loss, _, n_imgs = self.trainer.train_step(epoch=idx, feed_dict=feed_dict, loss_scale=loss_scale,
                                                learning_rate=self.adaptation_learning_rate)
      #print >> log.v4, "adapting on frame", adaption_frame_idx, "of sequence", video_idx + 1, \
      #         self.train_data.video_tag(video_idx), "loss:", loss
    return loss
