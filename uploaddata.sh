#!/bin/bash
MYDIR="$(dirname "$(realpath "$0")")"
cd ~/BoltDb/tf/big/
tar -cvzf data.tar.gz move.cvs.train* move.cvs.cv
scp  data.tar.gz rho@172.104.134.235:/home/rho/mypython/data/
rm data.tar.gz
cd $MYDIR
