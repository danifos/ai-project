#!/bin/bash

file=`date +%y%m%H%M`'.log'
split='============================================================'

echo 'Experiments log' > $file
date >> $file
echo $split >> $file

echo 'DNN' >> $file
python -u main.py --arch 100 200 400 400 --epochs=40 --bs 200 | tee -a $file
echo $split >> $file

