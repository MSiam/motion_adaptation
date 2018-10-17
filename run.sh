#USAGE : python main.py config_path video_id [ number of video or -1 to load all vids]

## DAVIS
### Adaptation
#python main.py configs/DAVIS16_unsupervised -1
#python main.py configs/DAVIS16_teach -1
#python main.py configs/DAVIS16_teachcont -1

### CRF Postprocessing
#python crf/crf_davis.py DAVIS16_unsupervised /home/mennatul/Datasets/DAVIS/
#python crf/crf_davis.py DAVIS16_online_cont /home/nray1/ms/DAVIS/
#python crf/crf_davis.py DAVIS16_online /home/nray1/ms/DAVIS/

## LORDS HRI
### Scale or Rotation
#python main.py configs/fordsm_unsupervised -1
#python main.py configs/fordsm_teach -1
python main.py configs/fordsm_teachcont -1

### Tasks on Bottles
#python main.py configs/fordsm_tasks_teach 0
#python main.py configs/fordsm_tasks_teachcont 0

## FBMS
#python main.py configs/fbms_teach -1
