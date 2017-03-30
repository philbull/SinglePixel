#!/bin/bash

# Check that enough arguments were specified
if [ "$#" -eq "0" ]
then
    echo "Needs 1 argument: model_in"
    exit 1
fi

# Get input model name
mod_in=$1

# Loop over random seeds
for i in `seq 100 200`
do 
    /usr/lib64/openmpi/bin/mpirun -n 36 python run_joint_mcmc.py $i \
                    synch,freefree,ame,$mod_in \
                    synch,freefree,ame,mbb
    echo $mod_in $i
done
