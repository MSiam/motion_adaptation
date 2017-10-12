HOME_DIR="/data/menna/DAVIS/"
for d in $(ls $HOME_DIR"Motion/"); do
    echo item: $d
    python nlc_online.py -imdir $HOME_DIR"JPEGImages/480p/"$d"/" -out $HOME_DIR"/Motion/" -maxsp 400 -iters 100
done
#python nlc_online.py -imdir $HOME_DIR"JPEGImages/480p/bmx-trees/" -out $HOME_DIR"/Motion/" -maxsp 400 -iters 100
