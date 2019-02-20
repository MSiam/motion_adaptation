# Video Segmentation using Teacher-Student Adaptation in a Human Robot Interaction (HRI) Setting.
The official implementation used in our [paper](https://arxiv.org/abs/1810.07733), in ICRA'19.

Video object segmentation is an essential task in robot manipulation to facilitate grasping and learning affordances. Incremental learning is important for robotics in unstructured environments, since the total number of objects and their variations can be intractable. Inspired by the children learning process, human robot interaction (HRI) can be utilized to teach robots about the world guided by humans similar to how children learn from a parent or a teacher. A human teacher can
show potential objects of interest to the robot, which is able to self adapt to the teaching signal without providing manual segmentation labels. We propose a novel teacher-student learning paradigm to teach robots about their surrounding environment. A two-stream motion and appearance "teacher" network provides pseudo-labels to adapt an appearance "student" network. The student network is able to segment the newly learned objects in other scenes, whether they are static or in
motion. We also introduce a carefully designed dataset that serves the proposed HRI setup, denoted as (I)nteractive (V)ideo (O)bject (S)egmentation. Our IVOS dataset contains teaching videos of different objects, and manipulation tasks. Unlike previous datasets, IVOS provides manipulation tasks sequences with segmentation annotation along with the waypoints for the robot trajectories. It also provides segmentation annotation for the different transformations such as translation, scale,
planar rotation, and out-of-plane rotation. Our proposed adaptation method outperforms the state-of-the-art on DAVIS and FBMS with 6.8% and 1.2% in F-measure respectively. It improves over the baseline on IVOS dataset with 46.1% and 25.9% in mIoU. 

<div align="center">
<img src="https://github.com/MSiam/motion_adaptation/blob/master/figures/overview.png" width="70%" height="70%"><br><br>
</div>

This implementation is based on the semi-supervised video segmentation method OnaVos implementation.
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

## IVOS Dataset

<div align="center">
<img src="https://github.com/MSiam/motion_adaptation/blob/master/figures/dataset1.png" width="70%" height="70%"><br><br>
</div>

<div align="center">
<img src="https://github.com/MSiam/motion_adaptation/blob/master/figures/dataset2.png" width="70%" height="70%"><br><br>
</div>


[Download Data Formatted for MotAdapt](https://drive.google.com/open?id=1lq2q-ivxppfbefj9pryVDK3NEvbjTnn1)

This work is a continuation on Team Alberta KUKA Innovation Award submission.
[Video Demo KUKA](https://www.youtube.com/watch?v=aLcw73dt_Oo)
[Video Demo for Our Paper](https://www.youtube.com/watch?v=36hMbAs8e0c&t=17s)

[IVOS Dataset](https://msiam.github.io/ivos/)

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
