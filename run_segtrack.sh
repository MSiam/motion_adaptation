#for i in `seq 0 13`;
#do
#    python main.py configs/DAVIS16_online $i
#done
python main.py configs/DAVIS16_online 6
python main.py configs/DAVIS16_online 8
python main.py configs/DAVIS16_online 10
python main.py configs/DAVIS16_online 12
