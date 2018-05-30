#!/bin/bash
source $PATH/test.sh
source /fefs/opt/dgx/env_set/nvcr-tensorflow-1712.sh
time python $PATH/fm.py --dataset $PATH/data/pfa_$dataset --logistic >> $PATH/$prefix-lr.txt 2>> $PATH/$prefix-lr.err
time python $PATH/fm.py --dataset $PATH/data/pfa_$dataset --iter $fm_iter --d $d >> $PATH/$prefix-fm0.txt 2>> $PATH/$prefix-fm0.err
time python $PATH/dfm.py --dataset $PATH/data/pfa_$dataset --iter $iter --d $d --rate $rate --fm >> $PATH/$prefix-fm.txt 2>> $PATH/$prefix-fm.err
time python $PATH/dfm.py --dataset $PATH/data/pfa_$dataset --iter $iter --d $d --rate $rate --deep >> $PATH/$prefix-deep.txt 2>> $PATH/$prefix-deep.err
time python $PATH/dfm.py --dataset $PATH/data/pfa_$dataset --iter $iter --d $d --rate $rate --deep --fm >> $PATH/$prefix-deepfm.txt 2>> $PATH/$prefix-deepfm.err
