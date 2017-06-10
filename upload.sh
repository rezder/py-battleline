#!/bin/bash
MYDIR="$(dirname "$(realpath "$0")")"
cd ~/Python/tensorflow/battleline/
tar -cvzf battup.tar doms.py posreader.py batt.py
mv battup.tar $MYDIR
cd $MYDIR
scp  battup.tar.gz rho@172.104.155.20:/home/rho/mypython/
rm battup.tar.gz
