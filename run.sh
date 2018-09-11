#USAGE : python main.py config_path video_id [ number of video or -1 to load all vids]

#for i in `seq 0 32`;
#do
#    python main.py configs/fordsm_online $i
#done
#python main.py configs/fordsm_online 0

#python main.py configs/DAVIS16_unsupervised -1
#python main.py configs/DAVIS16_teach -1

#python crf/crf_davis.py DAVIS16_unsupervised /home/nray1/ms/DAVIS/
#python crf/crf_davis.py DAVIS16_online_cont /home/nray1/ms/DAVIS/
#python crf/crf_davis.py DAVIS16_online /home/nray1/ms/DAVIS/

#python main.py configs/fordsm_unsupervised -1
#python main.py configs/fordsm_teachcont -1
#python main.py configs/DAVIS16_teachcont_refined -1
#python main.py configs/fordsm_tasks_teach 0
#python main.py configs/fordsm_tasks_teachcont 0
python main.py configs/fbms_teach -1

#python crf/crf_davis.py FORDS_online /home/nray1/ms/FORDS_Scale/


