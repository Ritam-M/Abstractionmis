#!/bin/sh

# have job exit if any command returns with non-zero exit status (aka failure)
set -e

# 1. setup anaconda environment
# replace env-name on the right hand side of this line with the name of your conda environment
ENVNAME=research
# if you need the environment directory to be named something other than the environment name, change this line
ENVDIR=$ENVNAME

# these lines handle setting up the environment; you shouldn't have to modify them
export PATH
mkdir $ENVDIR
tar -xzf $ENVNAME.tar.gz -C $ENVDIR
. $ENVDIR/bin/activate

sleep 5

# launch code
python3 run_single_dis.py "$@"

