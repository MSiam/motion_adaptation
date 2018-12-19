# Video Segmentation using Teacher-Student Adaptation in a Human Robot Interaction (HRI) Setting.
The official implementation used in our [paper](https://arxiv.org/abs/1810.07733):


<div align="center">
<img src="https://github.com/MSiam/motion_adaptation/blob/master/figures/overview.png" width="70%" height="70%"><br><br>
</div>

This implementation is based on the semi-supervised video segmentation method OnaVos.
[Onavos](https://www.vision.rwth-aachen.de/page/OnAVOS)

## Installation

```
virtualenv --system-site-packages ~/.tfenvpy2
source ~/.tfenvpy2/bin/activate
pip install tensorflow-gpu
pip install sklearn
pip install scikit-image
./run.sh
```

### Example Usage
To run on a certain sequence
```
python main.py CONFIG_FILE VID_ID
```
* CONFIG_FILE - Configuration File path
* VID_ID - integer denoting the video index, if -1 all videos are used.

### Configuration File

Explanation of the most important config parameters:

* targets_path: Path to folder with adaptation targets
* few_shot_samples: Number of samples to use in the adaptation.
* n_adaptation_steps: number of update steps per sample.
* adaptation_learning_rate: learning rate used during online adaptation
* adapt_th: threshold used to obtain the positive training examples
* distance_negative_threshold: distance threshold used to select the negative examples

## LORDS-HRI Dataset
A newer version of the dataset with the refined and full segmentation annotations will soon be released.

<div align="center">
<img src="https://github.com/MSiam/motion_adaptation/blob/master/figures/dataset1.png" width="70%" height="70%"><br><br>
</div>

<div align="center">
<img src="https://github.com/MSiam/motion_adaptation/blob/master/figures/dataset2.png" width="70%" height="70%"><br><br>
</div>


[Download Link](https://drive.google.com/file/d/1lq2q-ivxppfbefj9pryVDK3NEvbjTnn1/view?usp=sharing)

This work is a continuation on Team Alberta KUKA Innovation Award submission.
[Video Demo KUKA](https://www.youtube.com/watch?v=aLcw73dt_Oo)

Dataset webpage under construction here:
[LORDS-HRI Dataset](https://msiam.github.io/lordshri/)

## Cite
Please cite our work if you use the dataset or method outlined:
@article{siam2018video,
  title={Video Segmentation using Teacher-Student Adaptation in a Human Robot Interaction (HRI) Setting},
  author={Siam, Mennatullah and Jiang, Chen and Lu, Steven and Petrich, Laura and Gamal, Mahmoud and Elhoseiny, Mohamed and Jagersand, Martin},
  journal={arXiv preprint arXiv:1810.07733},
  year={2018}
}
