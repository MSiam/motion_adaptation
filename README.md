# Video Segmentation using Teacher-Student Adaptation in a Human Robot Interaction (HRI) Setting.
The official implementation used in our [paper](https://arxiv.org/abs/1810.07733).

Video segmentation is a challenging task that has many applications in robotics. Learning segmentation from few examples on-line is important for robotics in unstructured environments. The total number of objects and their variation in the real world is intractable, but for a specific task the robot deals with a small subset. Our network is taught, by a human moving a hand-held object through different poses. A novel two-stream motion and appearance "teacher" network provides pseudo-labels. These labels are used to adapt an appearance "student" network. Segmentation can be used to support a variety of robot vision functionality, such as grasping or affordance segmentation. We propose different variants of motion adaptation training and extensively compare against the state-of-the-art methods. We collected a carefully designed dataset in the human robot interaction (HRI) setting. We denote our dataset as (L)ow-shot (O)bject (R)ecognition, (D)etection and (S)egmentation using HRI. Our dataset contains teaching videos of different hand-held objects moving in translation, scale and rotation. It contains kitchen manipulation tasks as well, performed by humans and robots. Our proposed method outperforms the state-of-the-art on DAVIS and FBMS with 7% and 1.2% in F-measure respectively. In our more challenging LORDS-HRI dataset, our approach achieves significantly better performance with 46.7% and 24.2% relative improvement in mIoU over the baseline. 

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
```
@article{siam2018video,
  title={Video Segmentation using Teacher-Student Adaptation in a Human Robot Interaction (HRI) Setting},
  author={Siam, Mennatullah and Jiang, Chen and Lu, Steven and Petrich, Laura and Gamal, Mahmoud and Elhoseiny, Mohamed and Jagersand, Martin},
  journal={arXiv preprint arXiv:1810.07733},
  year={2018}
}
```
