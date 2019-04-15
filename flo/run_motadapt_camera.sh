rm -rf teaching
source ../../../.pyenv2/bin/activate
python -m src.flownet2.test --out_path teaching/ --cam_idx 0

cd ../../Mask_RCNN/samples/
source ../../../.tf3venv/bin/activate
python run_video.py ../../motion_adaptation/flownet2tf/teaching/

cd ../../motion_adaptation/
source ../../.pyenv2/bin/activate
python main.py configs/DAVIS16_2stream_unsupervised 0 0

python main.py configs/DAVIS16_teach 0 0
