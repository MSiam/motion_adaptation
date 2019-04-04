import argparse
import os
from ..net import Mode
from .flownet2 import FlowNet2
import cv2
import os

FLAGS = None


def save_imgs(img, flow, out_path, idx):
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    cv2.imwrite(out_path + '%05d_img.png'%idx, img)
    cv2.imwrite(out_path + '%05d_flo.png'%idx, flow)

def main(args):
    # Create a new network
    net = FlowNet2(mode=Mode.TEST)
    cap = cv2.VideoCapture(0)

    frame = None
    flowFlag = False
    saveFlag = False
    idx = 0

    flo_w = 512; flo_h = 384
    net.prepare_test(checkpoint='./checkpoints/FlowNet2/flownet-2.ckpt-0',
                     inp_size=(1, flo_h, flo_w, 3))

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
                save_imgs(frame, flow_img, args.out_path, idx)
                idx += 1

if __name__ == '__main__':
     parser = argparse.ArgumentParser()
     parser.add_argument(
        '--out_path',
        default='./teaching/',
        help='output path to save images used for teaching'
        )

     main(parser.parse_args())
