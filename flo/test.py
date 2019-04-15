import argparse
import os
from ..net import Mode
from .flownet2 import FlowNet2
import cv2
import os
from Tkinter import *
import threading
import PIL.Image, PIL.ImageTk

FLAGS = None

class Interface(object):
    def __init__(self, args):
        self.args = args
        self.out_path = self.args.out_path

        # Create a new network
        self.net = FlowNet2(mode=Mode.TEST)
        self.cap = cv2.VideoCapture(self.args.cam_idx)

        self.frame = None
        self.prev_frame = None
        self.saveFlag = False
        self.idx = 0

        self.flo_w = 512; self.flo_h = 384
        self.net.prepare_test(checkpoint='./checkpoints/FlowNet2/flownet-2.ckpt-0',
                              inp_size=(1, self.flo_h, self.flo_w, 3))

        self.panel_img = None
        self.panel_of = None

        self.sq_idx = ''

    def save_imgs(self, img, flow, out_path):
        cv2.imwrite(out_path + 'JPEGImages/sq_' + \
                str(self.sq_idx) + '/%05d.png'%self.idx, img)
        cv2.imwrite(out_path + 'OpticalFlow/sq_' + \
                str(self.sq_idx) + '/%05d.png'%self.idx, flow)

    def set_panel(self, panel, photo, r, c):
        if panel is None:
            panel = Label(self.root, image=photo)#Canvas(self.root, width = self.flo_w, height = self.flo_h)
            panel.image = photo
            #panel.pack(side=side, padx=10, pady=10)
            panel.grid(row=r, column=c, columnspan=2)
        else:
            panel.configure(image=photo)
            panel.image = photo
        return panel

    def videoLoop(self):
        while not self.stopEvent.is_set():
            self.prev_frame = self.frame
            _, self.frame = self.cap.read()

            if self.prev_frame is not None:
                self.prev_frame = cv2.resize(self.prev_frame,
                                             (self.flo_w, self.flo_h))
            self.frame = cv2.resize(self.frame,
                                    (self.flo_w, self.flo_h))
            photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.frame[:,:,::-1]))
            self.panel_img = self.set_panel(self.panel_img, photo, 0, 0)

            if self.prev_frame is not None:
                flow_img = self.net.test(
                                    input_a=self.prev_frame,
                                    input_b=self.frame,
                                    out_path='',
                                    save_image=False
                                   )

                photo_of = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(flow_img[:,:,::-1]))
                self.panel_of = self.set_panel(self.panel_of, photo_of, 0, 2)

                if self.saveFlag:
                    self.save_imgs(self.frame, flow_img, self.out_path)
                    self.f.write('JPEGImages/sq_' + str(self.sq_idx) + '/%05d.png '%self.idx +\
                            'Annotations/sq_' + str(self.sq_idx) + '/%05d.png\n'%self.idx)
                    self.idx += 1

    def create_directories(self):
        if not os.path.exists(self.out_path):
            os.mkdir(self.out_path)
        if not os.path.exists(self.out_path + 'JPEGImages/'):
            os.mkdir(self.out_path + 'JPEGImages/')
        if not os.path.exists(self.out_path + 'JPEGImages/sq_' + self.sq_idx):
            os.mkdir(self.out_path + 'JPEGImages/sq_' + str(self.sq_idx) + '/')
        if not os.path.exists(self.out_path + 'OpticalFlow/'):
            os.mkdir(self.out_path + 'OpticalFlow/')
        if not os.path.exists(self.out_path + 'OpticalFlow/sq_' + self.sq_idx):
            os.mkdir(self.out_path + 'OpticalFlow/sq_' + str(self.sq_idx) + '/')
        if not os.path.exists(self.out_path + 'ImageSets/'):
            os.mkdir(self.out_path + 'ImageSets/')
            os.mkdir(self.out_path + 'ImageSets/480p')

        if os.path.exists(self.out_path + 'ImageSets/480p/val.txt'):
            self.f = open(self.out_path + 'ImageSets/480p/val.txt', 'w')
        else:
            self.f = open(self.out_path + 'ImageSets/480p/val.txt', 'a')

    def onClose(self):
	self.stopEvent.set()
	self.root.quit()
        self.f.close()

    def onSave(self):
        self.sq_idx = self.cls_txt.get()
        self.create_directories()
        self.saveFlag = True

    def onDone(self):
        self.saveFlag = False
        self.idx = 0

    def main(self):
        self.root = Tk()

        labelfont = ('times', 15, 'bold')

        self.cls_txt = StringVar()
        cls = Entry(self.root, textvariable=self.cls_txt)
        cls.config(font=labelfont)
        cls.grid(row=1, column=1)

        cls_label = Label(self.root, text="Class Name")
        cls_label.config(font=labelfont)
        cls_label.grid(row=1, column=0)

        self.save_btn = Button(self.root, text='Save', command=self.onSave)
        self.save_btn.config(font=labelfont)
        self.save_btn.grid(row=1, column=2)

        self.done_btn = Button(self.root, text='Done', command=self.onDone)
        self.done_btn.config(font=labelfont)
        self.done_btn.grid(row=1, column=3)

        self.stopEvent = threading.Event()
        thread = threading.Thread(target=self.videoLoop, args=())
        thread.start()

        self.root.wm_title("MotAdapt Teaching Phase")
        self.root.wm_protocol("WM_DELETE_WINDOW", self.onClose)

        self.root.mainloop()

if __name__ == '__main__':
     parser = argparse.ArgumentParser()
     parser.add_argument(
        '--out_path',
        default='./teaching/',
        help='output path to save images used for teaching'
        )
     parser.add_argument(
         '--cam_idx',
         default=0,
         type=int,
         help='number of camera used in Video Capture'
         )
     gui = Interface(parser.parse_args())
     gui.main()
