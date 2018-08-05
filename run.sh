#USAGE : python main.py config_path video_id [ number of video or -1 to load all vids]

#for i in `seq 0 32`;
#do
#    python main.py configs/custom_online $i
#done
#python main.py configs/custom_online 0
python main.py configs/DAVIS16_unsupervised -1


