"""
This file implements following paper:
Video Segmentation by Non-Local Consensus Voting
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from __future__ import unicode_literals
import os
import sys
from PIL import Image
import numpy as np
from skimage.segmentation import slic
from skimage.feature import hog
from skimage import color
from scipy.spatial import KDTree
from scipy.misc import imresize
from scipy import ndimage
# from cv2 import calcOpticalFlowFarneback, OPTFLOW_FARNEBACK_GAUSSIAN
from scipy.signal import convolve2d
import time
import utils
import _init_paths  # noqa
from mr_saliency import MR
import pyflow
import scipy
import scipy.stats
import pdb
import matplotlib.pyplot as plt

def compute_saliency(imSeq, flowSz=100, flowBdd=12.5, flowF=3, flowWinSz=10,
                        flowMagTh=1, flowDirTh=0.75, numDomFTh=0.5,
                        flowDirBins=10, patchSz=5, redirect=False,
                        doNormalize=True, defaultToAppearance=True):
    """
    Initialize for FG/BG votes by Motion or Appearance Saliency. FG>0, BG=0.
    Input:
        imSeq: (n, h, w, c) where n > 1: 0-255: np.uint8: RGB
        flowSz: target size of image to be resized to for computing optical flow
        flowBdd: percentage of smaller side to be removed from bdry for saliency
        flowF: temporal radius to find optical flow
        flowWinSz: winSize in farneback (large -> get fast motion, but blurred)
        numDomFTh: # of dominant frames needed for motion Ssliency
        flowDirBins: # of bins in flow orientation histogram
        patchSz: patchSize for obtaining motion saliency score
    Output:
        salImSeq: (n, h, w) where n > 1: float. FG>0, BG=0. score in [0,1].
    """

    def isDominant(flow, flowMagTh, flowDirTh, dirBins=10):
        mag = np.square(flow)
        mag = np.sqrt(mag[..., 0] + mag[..., 1])
        med = np.median(mag)
        dominant = False
        target = -1000
        moType = ''
        if med < flowMagTh:
            dominant = True
            targetIm = mag
            target = 0.
            moType = 'static'

        if not dominant:
            # orientation in radians: (-pi, pi): disambiguates sign of arctan
            orien = np.arctan2(flow[..., 1], flow[..., 0])
            # use ranges, number of bins and normalization to compute histogram
            dirHist, bins = np.histogram(orien, bins=dirBins, weights=mag,
                                            range=(-np.pi, np.pi))
            dirHist /= np.sum(dirHist) + (np.sum(dirHist) == 0)

            #plt.bar(bins[:-1], dirHist, width=np.diff(bins), ec="k", align="edge")
            #plt.show()

            if np.max(dirHist) > flowDirTh:
                dominant = True
                targetIm = orien
                target = bins[np.argmax(dirHist)] + bins[np.argmax(dirHist) + 1]
                target /= 2.
                moType = 'translate'

        if dominant:
            # E[(x-mu)^2]
            deviation = (targetIm - target)**2
            if moType == 'translate':
                # for orientation: theta = theta + 2pi. Thus, we want min of:
                # (theta1-theta2) = (theta1-theta2-2pi) = (2pi+theta1-theta2)
                deviation = np.minimum(
                    deviation, (targetIm - target + 2. * np.pi)**2)
                deviation = np.minimum(
                    deviation, (targetIm - target - 2. * np.pi)**2)
            saliency = convolve2d(
                deviation, np.ones((patchSz, patchSz)) / patchSz**2,
                mode='same', boundary='symm')
            return dominant, moType, target, saliency

        return dominant, moType, target, -1000

    sTime = time.time()
    # pyflow Options:
    alpha = 0.012
    ratio = 0.75
    minWidth = 20
    nOuterFPIterations = 7
    nInnerFPIterations = 1
    nSORIterations = 30

    n, h, w, c = imSeq.shape
    im = np.zeros((n, flowSz, flowSz, c), np.uint8)

    # decrease size for optical flow computation
    for i in range(n):
        im[i] = imresize(imSeq[i], (flowSz, flowSz))

    # compute Motion Saliency per frame
    valid= True
    confs= []
    salImSeq = np.zeros((n, flowSz, flowSz))
    numDomFrames = 0
    motion_types=[]
    for i in range(n):
        isFrameDominant = 0
        for j in range(-flowF, flowF + 1):
            if j == 0 or i + j < 0 or i + j >= n:
                continue
            # flow = calcOpticalFlowFarneback(
            #     color.rgb2gray(im[i]), color.rgb2gray(im[i + j]), 0.5, 4,
            #     flowWinSz, 10, 5, 1.1, OPTFLOW_FARNEBACK_GAUSSIAN)
            # pyflow needs im: float in [0,1]
            u, v, _ = pyflow.coarse2fine_flow(
                im[i].astype(float) / 255., im[i + j].astype(float) / 255.,
                alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
                nSORIterations, 0)
            flow = np.concatenate((u[..., None], v[..., None]), axis=2)

            dominant, motype, target, salIm = isDominant(
                flow, flowMagTh, flowDirTh, dirBins=flowDirBins)
            motion_types+= [motype]

            if False:
                odir = '/home/dpathak/local/data/trash/my_nlc/nlc_out/'
                np.save(odir + '/np/outFlow_%d_%d.npy' % (i, i + j), flow)
                import cv2
                hsv = np.zeros((100, 100, 3), dtype=np.uint8)
                hsv[:, :, 0] = 255
                hsv[:, :, 1] = 255
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                hsv[..., 0] = ang * 180 / np.pi / 2
                hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                cv2.imwrite(odir + '/im/outFlow_%d_%d.png' % (i, i + j), rgb)

            if dominant:
                salImSeq[i] += salIm
                isFrameDominant += 1

        if isFrameDominant > 0:
            salImSeq[i] /= isFrameDominant
            numDomFrames += isFrameDominant > 0
            confs.append(True)
        else:
            confs.append(False)

        if not redirect:
            sys.stdout.write('Motion Saliency computation: [% 5.1f%%]\r' %
                                (100.0 * float((i + 1) / n)))
            sys.stdout.flush()
    eTime = time.time()
    print('Motion Saliency computation finished: %.2f s' % (eTime - sTime))
    print('Number of dominant frames is ', numDomFrames,' th= ', numDomFTh)
    print('confidences ', len( np.where(confs==True)[0]) )
    print('Motion Types ', motion_types)
    sq_motion_type= scipy.stats.mode(motion_types)[0][0]
    if numDomFrames < n * numDomFTh and defaultToAppearance:
        valid= False
        sq_motion_type= 'rotate'

    if sq_motion_type== '' and valid:
        sq_motion_type= 'translate'
    # resize back to image size, and exclude boundaries
    exclude = int(min(h, w) * flowBdd * 0.01)
    salImSeqOrig = np.zeros((n, h, w))
    for i in range(n):
        # bilinear interpolation to upsample back
        salImSeqOrig[i, exclude:-exclude, exclude:-exclude] = \
            ndimage.interpolation.zoom(
            salImSeq[i], (h * 1. / flowSz, w * 1. / flowSz), order=1)[
            exclude:-exclude, exclude:-exclude]

    # normalize full video, and NOT per frame
    if np.max(salImSeqOrig) > 0 and doNormalize:
        salImSeqOrig /= np.max(salImSeqOrig)

    return salImSeqOrig, confs, valid, sq_motion_type


def salScore2votes(salImSeq, sp):
    """
    Convert saliency score to votes
    Input:
        salImSeq: (n, h, w) where n > 1: float. FG>0, BG=0. score in [0,1].
        sp: (n,h,w): 0-indexed regions, #regions <= numsp
    Output:
        votes: (k,) where k < numsp*n
    """
    n, h, w = salImSeq.shape
    numsp = np.max(sp) + 1
    votes = np.zeros((numsp * n,), dtype=np.float)
    startInd = 0
    for i in range(n):
        sp1 = sp[i].reshape(-1)
        val1 = salImSeq[i].reshape(-1)
        sizeOut = np.max(sp1) + 1
        # assign average score of pixels to a superpixel
        sumScore = utils.my_accumarray(sp1, val1, sizeOut, 'plus')
        count = utils.my_accumarray(sp1, np.ones(sp1.shape), sizeOut, 'plus')
        votes[startInd:startInd + sizeOut] = sumScore / count
        startInd += sizeOut
    votes = votes[:startInd]

    return votes


def consensus_vote(votes, transM, frameEnd, iters):
    """
    Perform iterative consensus voting
    """
    sTime = time.time()
    for t in range(iters):
        votes = np.dot(transM, votes)
        # normalize per frame
        for i in range(frameEnd.shape[0]):
            currStartF = 1 + frameEnd[i - 1] if i > 0 else 0
            currEndF = frameEnd[i]
            frameVotes = np.max(votes[currStartF:1 + currEndF])
            votes[currStartF:1 + currEndF] /= frameVotes + (frameVotes <= 0)
    eTime = time.time()
    print('Consensus voting finished: %.2f s' % (eTime - sTime))
    return votes


def votes2mask(votes, sp):
    """
    Project votes to images to obtain masks
    Input:
        votes: (k,) where k < numsp*n
        sp: (h,w) or (n,h,w): 0-indexed regions, #regions <= numsp
    Output:
        maskSeq: (h,w) or (n,h,w):float. FG>0, BG=0.
    """
    if sp.ndim < 3:
        sp = sp[None, ...]

    # operation is inverse of accumarray, i.e. indexing
    n, h, w = sp.shape
    maskSeq = np.zeros((n, h, w))
    startInd = 0
    for i in range(n):
        sp1 = sp[i].reshape(-1)
        sizeOut = np.max(sp1) + 1
        voteIm = votes[startInd:startInd + sizeOut]
        maskSeq[i] = voteIm[sp1].reshape(h, w)
        startInd += sizeOut

    if sp.ndim < 3:
        return maskSeq[0]
    return maskSeq


def remove_low_energy_blobs(maskSeq, binTh, relSize=0.6, relEnergy=None,
                                target=None):
    """
    Input:
        maskSeq: (n, h, w) where n > 1: float. FG>0, BG=0. Not thresholded.
        binTh: binary threshold for maskSeq for finding blobs: [0, max(maskSeq)]
        relSize: [0,1]: size of FG blobs to keep compared to largest one
                        Only used if relEnergy is None.
        relEnergy: Ideally it should be <= binTh. Kill blobs whose:
                    (total energy <= relEnergy * numPixlesInBlob)
                   If relEnergy is given, relSize is not used.
        target: value to which set the low energy blobs to.
                Default: binTh-epsilon. Must be less than binTh.
    Output:
        maskSeq: (n, h, w) where n > 1: float. FG>0, BG=0. Not thresholded. It
                 has same values as input, except the low energy blobs where its
                 value is target.
    """
    sTime = time.time()
    if target is None:
        target = binTh - 1e-5
    for i in range(maskSeq.shape[0]):
        mask = (maskSeq[i] > binTh).astype(np.uint8)
        if np.sum(mask) == 0:
            continue
        sp1, num = ndimage.label(mask)  # 0 in sp1 is same as 0 in mask i.e. BG
        count = utils.my_accumarray(sp1, np.ones(sp1.shape), num + 1, 'plus')
        if relEnergy is not None:
            sumScore = utils.my_accumarray(sp1, maskSeq[i], num + 1, 'plus')
            destroyFG = sumScore[1:] < relEnergy * count[1:]
        else:
            sizeLargestBlob = np.max(count[1:])
            destroyFG = count[1:] < relSize * sizeLargestBlob
        destroyFG = np.concatenate(([False], destroyFG))
        maskSeq[i][destroyFG[sp1]] = target
    eTime = time.time()
    print('Removing low energy blobs finished: %.2f s' % (eTime - sTime))
    return maskSeq

def parse_args():
    """
    Parse input arguments
    """
    import argparse
    parser = argparse.ArgumentParser(
        description='Foreground Segmentation using Non-Local Consensus')
    parser.add_argument(
        '-out', dest='outdir',
        help='Directory to save output.',
        default=os.getenv("HOME") + '/local/data/trash/', type=str)
    parser.add_argument(
        '-imdir', dest='imdir',
        help='Directory containing video images. Will be read ' +
        'alphabetically. Default is random Imagenet train video.',
        default='', type=str)
    parser.add_argument(
        '-fgap', dest='frameGap',
        help='Gap between frames while running tracker. Default 0.',
        default=0, type=int)
    parser.add_argument(
        '-maxsp', dest='maxsp',
        help='Max # of superpixels per image. Default 0.',
        default=1000, type=int)
    parser.add_argument(
        '-iters', dest='iters',
        help='# of iterations of consensus voting. Default 100.',
        default=100, type=int)
    parser.add_argument(
        '-seed', dest='seed',
        help='Random seed for numpy and python.', default=2905, type=int)

    args = parser.parse_args()
    return args


def demo_images():
    """
    Input is the path of directory (imdir) containing images of a video
    """
    # Hard coded parameters
    maxSide = 600  # max length of longer side of Im
    lenSeq = 35  # longer seq will be shrinked between [lenSeq/2, lenSeq]
    binTh = 0.4  # final thresholding to obtain mask
    clearFinalBlobs = True  # remove low energy blobs; uses binTh

    # parse commandline parameters
    args = parse_args()
    np.random.seed(args.seed)
    # setup input directory
    print('InputDir: ', args.imdir)
    imPathList = utils.read_r(args.imdir, '*.*')
    if len(imPathList) < 2:
        print('Not enough images in image directory: \n%s' % args.imdir)
        return

    # setup output directory
    suffix = args.imdir.split('/')[-1]
    suffix = args.imdir.split('/')[-2] if suffix == '' else suffix
    args.outdir = args.outdir + '/' + suffix
    utils.mkdir_p(args.outdir)
    print('OutputDir: ', args.outdir)

    # load image sequence after adjusting frame gap and imsize
    frameGap = args.frameGap
    if frameGap <= 0 and len(imPathList) > lenSeq:
        frameGap = int(len(imPathList) / lenSeq)
    indices= range(0,len(imPathList),frameGap+1)
    imPathList = imPathList[0:len(imPathList):frameGap + 1]
    h, w, c = np.array(Image.open(imPathList[0])).shape
    #frac = min(min(1. * maxSide / h, 1. * maxSide / w), 1.0)
    #if frac < 1.0:
    #    h, w, c = imresize(np.array(Image.open(imPathList[0])), frac).shape
    imSeq = np.zeros((len(imPathList), h, w, c), dtype=np.uint8)
    for i in range(len(imPathList)):
        #if frac < 1.0:
        #    imSeq[i] = imresize(np.array(Image.open(imPathList[i])), frac)
        #else:
        imSeq[i] = np.array(Image.open(imPathList[i]))

    print('Total Video Shape: ', imSeq.shape)

    # run the algorithm
    salImSeq, confs, valid, motype = compute_saliency(imSeq, flowBdd=12.5, flowDirBins=20,
                                        redirect=False)

    print('Dominant motion type is ', motype )
    maskSeq= salImSeq
    #maskSeq, confs, valid = nlc(imSeq, maxsp=args.maxsp, iters=args.iters, outdir=args.outdir, dosave=False)
    np.save(args.outdir + '/mask_%s.npy' % suffix, maskSeq)

    # save visual results
    masks= []
    if clearFinalBlobs:
        maskSeq = remove_low_energy_blobs(maskSeq, binTh)
    utils.rmdir_f(args.outdir + '/result_%s/' % suffix)
    utils.mkdir_p(args.outdir + '/result_%s/' % suffix)
    for i in range(maskSeq.shape[0]):
        th= maskSeq[i].max()/2.0
        mask= (maskSeq[i]> th).astype(np.uint8)*255
        scipy.misc.imsave(args.outdir + '/result_%s/' % suffix + imPathList[i].split('/')[-1], mask)

#        mask = (maskSeq[i] > binTh).astype(np.uint8)
#        masks.append(mask)
#        grayscaleimage = (color.rgb2gray(imSeq[i]) * 255.).astype(np.uint8)
#        imMasked = np.zeros(imSeq[i].shape, dtype=np.uint8)
#        for c in range(3):
#            imMasked[:, :, c] = grayscaleimage / 2 + 127
#        imMasked[mask.astype(np.bool), 1:] = 0
#        Image.fromarray(imMasked).save()

    np.save(args.outdir+'/sal_th.npy', masks)
    np.save(args.outdir+'/confs.npy', confs)
    np.save(args.outdir+'/valid.npy', valid)
    np.save(args.outdir+'/indices.npy', indices)
    np.save(args.outdir+'/motype.npy', motype)
#    import subprocess
#    subprocess.call(
#        ['tar', '-zcf', args.outdir + '/../result_%s.tar.gz' % suffix,
#            '-C', args.outdir + '/result_%s/' % suffix, '.'])
    return


if __name__ == "__main__":
    # demo_videos()
    demo_images()
