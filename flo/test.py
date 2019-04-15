import argparse
import os
from ..net import Mode
from .flownet2 import FlowNet2
import cv2
import os

FLAGS = None


def save_imgs(img, flow, out_path, idx):
    cv2.imwrite(out_path + 'JPEGImages/sq1/%05d.png'%idx, img)
    cv2.imwrite(out_path + 'OpticalFlow/sq1/%05d.png'%idx, flow)

def main(args):
    # Create a new network
    net = FlowNet2(mode=Mode.TEST)
    cap = cv2.VideoCapture(args.cam_idx)

    frame = None
    flowFlag = False
    saveFlag = False
    idx = 0

    flo_w = 512; flo_h = 384
    net.prepare_test(checkpoint='./checkpoints/FlowNet2/flownet-2.ckpt-0',
                     inp_size=(1, flo_h, flo_w, 3))

    out_path = args.out_path
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    if not os.path.exists(out_path + 'JPEGIamges/'):
        os.mkdir(out_path + 'JPEGImages/')
        os.mkdir(out_path + 'JPEGImages/sq1/')
    if not os.path.exists(out_path + 'OpticalFlow/'):
        os.mkdir(out_path + 'OpticalFlow/')
        os.mkdir(out_path + 'OpticalFlow/sq1/')
    if not os.path.exists(out_path + 'ImageSets/'):
        os.mkdir(out_path + 'ImageSets/')
        os.mkdir(out_path + 'ImageSets/480p')

    if os.path.exists(out_path + 'ImageSets/480p/val.txt'):
        f = open(out_path + 'ImageSets/480p/val.txt', 'w')
    else:
        f = open(out_path + 'ImageSets/480p/val.txt', 'a')

    while True:
        prev_frame = frame
        ret, frame = cap.read()
        if prev_frame is not None:
            prev_frame = cv2.resize(prev_frame, (flo_w, flo_h))
        frame = cv2.resize(frame, (flo_w, flo_h))

        cv2.imshow('Live Feed', frame)
        ch = cv2.waitKey(10)%256

        if ch == ord('q'):
            break
        elif ch == ord('o'):
            flowFlag = not flowFlag
        elif ch == ord('s'):
            saveFlag = not saveFlag

        if flowFlag:
            flow_img = net.test(
                                input_a=prev_frame,
                                input_b=frame,
                                out_path='',
                                save_image=False
                               )
            cv2.imshow('Optical Flow ', flow_img)

            if saveFlag:
                save_imgs(frame, flow_img, out_path, idx)
                f.write('JPEGImages/sq1/%05d.png '%idx + 'Annotations/sq1/%05d.png\n'%idx)
                idx += 1

    f.close()

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
     main(parser.parse_args())
