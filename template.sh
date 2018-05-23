#!/bin/bash
source $PATH/test.sh
source /fefs/opt/dgx/env_set/nvcr-tensorflow-1712.sh
time python $PATH/test.py --dataset $PATH/data/last_$dataset --iter $iter --d $d --rate $rate --fm >> $PATH/$prefix-fm.txt 2>> $PATH/$prefix-fm.err
time python $PATH/test.py --dataset $PATH/data/last_$dataset --iter $iter --d $d --rate $rate --deep >> $PATH/$prefix-deep.txt 2>> $PATH/$prefix-deep.err
time python $PATH/test.py --dataset $PATH/data/last_$dataset --iter $iter --d $d --rate $rate --deep --fm >> $PATH/$prefix-deepfm.txt 2>> $PATH/$prefix-deepfm.err
