# progress bar + full screen to show the teaching phase
# kitchen setting motivate with different poses / recycling setting / 
# 

rm -rf teaching
source ../../../.pyenv2/bin/activate
python -m src.flownet2.test --out_path teaching/ --cam_idx 1

cd ../../Mask_RCNN/samples/
source ../../../.tf3venv/bin/activate
python run_video.py ../../motion_adaptation/flownet2tf/teaching/ 0

cd ../../motion_adaptation/
source ../../.pyenv2/bin/activate
python main.py configs/DAVIS16_2stream_unsupervised 0 1
