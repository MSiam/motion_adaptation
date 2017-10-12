## README Code for OnAVOS:
Requires a good GPU with at least 11GB memory (e.g. 1080 TI or TITAN X)

Instructions for running:
1) install tensorflow and possibly other required libraries using pip
2) download the models and put them in OnAVOS/models/
3) choose a config you want to run from configs/
I recommend to start with configs/DAVIS16_oneshot (which does the one-shot approach without adaptation on DAVIS 2016)
4) change the data directori(es) in the first lines of the config to your DAVIS path
5) run "python main.py configs/DAVIS16_oneshot" (or a different config)

Additional instructions for DAVIS2017:
1) download the lucid data and change the path in configs/DAVIS17_online to point to it
2) copy the txt files from OnAVOS/ImageSets2017_with_ids/ to the ImageSets folder of your DAVIS 2017 data
3) follow instructions from above using the configs/DAVIS17_online config

Instructions for running on a custom dataset (note that this is based on the implementation for DAVIS2017, so that your dataset needs to be converted to the folder structure - an alternative is to write custom code to load your dataset):
1) download the pascal or pascal_up model and put them in OnAVOS/models/
2) choose a config you want to run from configs/ 
I recommend to start with configs/custom_oneshot The custom_up_oneshot adds upsampling layers, which might improve the accuracy, but will increase runtime memory consumption. If you want to add online adaptation, please compare to the online configs for DAVIS2016
3) Put your dataset in OnAVOS/custom_dataset while retaining the folder structure of the example images. The structure has to be the same as DAVIS 2017
4) run "python main.py configs/custom_oneshot" (or custom_up_oneshot)

Explanation of the most important config parameters:
n_finetune_steps: the number of steps for fine-tuning on the first frame
learning_rates: dictionary mapping from step number to a learning rate
n_adaptation_steps: number of update steps per frame during adaptation
adaptation_interval: during online adaptation, each adaptation_interval steps, the current frame is used for updating, otherwise the first frame
adaptation_learning_rate: learning rate used during online adaptation
posterior_positive_threshold: posterior probability threshold used to obtain the positive training examples
distance_negative_threshold: distance threshold (to the last mask) used to select the negative examples
adaptation_loss_scale: weighting factor of loss during online adaptation
adaptation_erosion_size: erosion size used during online adaptation (use 1 to disable erosion)
n_test_samples: the number of random sampled augmented versions of the input image per frame used during testing. Reduce this, to make inference much faster at the cost of a little bit accuracy

Outputs:
Log files will be stored in logs/
The results will be stored in forwarded/

Note that the code slightly changed since we wrote the paper and that randomness is involved, so the results you obtain might be different from the results reported in the paper. 
If the results differ significantly, please let me know.

If you need additional config or model files or have questions, please write me a mail: voigtlaender@vision.rwth-aachen.de

If you find this code useful, please consider citing
Paul Voigtlaender and Bastian Leibe: Online Adaptation of Convolutional Neural Networks for Video Object Segmentation, BMVC 2017
Paul Voigtlaender and Bastian Leibe: Online Adaptation of Convolutional Neural Networks for the 2017 DAVIS Challenge on Video Object Segmentation, The 2017 DAVIS Challenge on Video Object Segmentation - CVPR Workshops


## README Code for UNLC Video Segmentation:
## Video Segmentation and tracking

Code for unsupervised bottom-up video motion segmentation. uNLC is a reimplementation of the NLC algorithm by Faktor and Irani, BMVC 2014, that removes the trained edge detector and makes numerous other modifications and simplifications. For additional details, see section 5.1 in the <a href="http://cs.berkeley.edu/~pathak/unsupervised_video/">paper</a>. This repository also contains code for a very simple video tracker which we developed.

This code was developed and is used in our [CVPR 2017 paper on Unsupervised Learning using unlabeled videos](http://cs.berkeley.edu/~pathak/unsupervised_video/). Github repository for our CVPR 17 paper is [here](https://github.com/pathak22/unsupervised-video). If you find this work useful in your research, please cite:

    @inproceedings{pathakCVPR17learning,
        Author = {Pathak, Deepak and Girshick, Ross and Doll\'{a}r,
                  Piotr and Darrell, Trevor and Hariharan, Bharath},
        Title = {Learning Features by Watching Objects Move},
        Booktitle = {Computer Vision and Pattern Recognition ({CVPR})},
        Year = {2017}
    }
<hr/>


**Video Segmentation** using low-level vision based unsupervised methods. It is largely inspired from Non-Local Consensus [Faktor and Irani, BMVC 2014] method, but removes all trained components. This segmentation method includes and make use of code for optical flow, motion saliency code, appearance saliency, superpixel and low-level descriptors.

**Video Tracking** code includes deepmatch followed by epic flow (or farneback) and then doing homography followed by bipartite matching to obtain foreground tracks.

### Setup

1. Install optical flow
  ```Shell
  cd videoseg/lib/
  git clone https://github.com/pathak22/pyflow.git
  cd pyflow/
  python setup.py build_ext -i
  python demo.py    # -viz option to visualize output
  ```

2. Install Dense CRF code
  ```Shell
  cd videoseg/lib/
  git clone https://github.com/lucasb-eyer/pydensecrf.git
  cd pydensecrf/
  python setup.py build_ext -i
  PYTHONPATH=.:$PYTHONPATH python examples/inference.py examples/im1.png examples/anno1.png examples/out_new1.png
  # compare out_new1.png and out1.png -- they should be same
  ```

3. Install appearance saliency
  ```Shell
  cd videoseg/lib/
  git clone https://github.com/ruanxiang/mr_saliency.git
  ```

4. Install kernel temporal segmentation code
  ```Shell
  # cd videoseg/lib/
  # wget http://pascal.inrialpes.fr/data2/potapov/med_summaries/kts_ver1.1.tar.gz
  # tar -zxvf kts_ver1.1.tar.gz && mv kts_ver1.1 kts
  # rm -f kts_ver1.1.tar.gz

  # Edit kts/cpd_nonlin.py to remove weave dependecy. Due to this change, we are shipping the library.
  # Included in videoseg/lib/kts/ . However, it is not a required change if you already have weave installed
  # (which is mostly present by default).
  ```

5. Convert them to modules
  ```Shell
  cd videoseg/lib/
  cp __init__.py mr_saliency/
  cp __init__.py kts/
  ```

6. Run temporal segmentation:
  ```Shell
  time python vid2shots.py -imdir /home/dpathak/local/data/trash/my_nlc/imseq/v21/ -out /home/dpathak/local/data/trash/my_nlc/nlc_out/
  ```

7. Run NLC segmentation:
  ```Shell
  cd videoseg/src/
  time python nlc.py -imdir /home/dpathak/local/data/trash/my_nlc/imseq/3_tmp/ -out /home/dpathak/local/data/trash/my_nlc/nlc_out/ -maxsp 400 -iters 100
  ```

8. Run Tracker:
  ```Shell
  cd videoseg/src/
  time python dm_tracker.py -fgap 2 -seed 2905 -vizTr -dmThresh 0 -shotFrac 0.2 -matchNbr 20 -postTrackHomTh -1 -preTrackHomTh 10
  ```

9. Run CRF sample:
  ```Shell
  cd videoseg/src/
  time python crf.py -inIm ../lib/pydensecrf/examples/im1.png -inL ../lib/pydensecrf/examples/anno1.png -out ../lib/pydensecrf/examples/out_new2.png
  ```

10. Run Full Pipeline:
  ```Shell
  cd videoseg/src/
  time python run_full.py -out /home/dpathak/local/data/AllSegInit -in /home/dpathak/fbcode/experimental/deeplearning/dpathak/videoseg/datasets/imdir_Samples.txt
  ```
