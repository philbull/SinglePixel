#!/bin/bash

# Check that enough arguments were specified
if [ "$#" -ne "2" ]
then
    echo "Needs 2 arguments: seed_range model_in"
    exit 1
fi

# Get seed range
seed_range=$1

# Get input model name
mod_in=$2

# Loop over random seeds
/usr/lib64/openmpi/bin/mpirun -n 60 python run_many_mcmc.py $seed_range \
                synch,freefree,ame,$mod_in \
                synch,freefree,ame,mbb
echo $seed_range $mod_in
