#!/bin/bash
source /etc/profile.d/lmod.sh
source /etc/profile.d/easybuild.sh
# source activate dlnn38

#source activate /homedtic/gmarti/ENV/dl38
module load Python
module load CUDA/10.2.89
module load cuDNN/7.6.5.32-CUDA-10.2.89
source /homedtic/gmarti/pytorch/bin/activate 
export XDG_RUNTIME_DIR=""
#srun --nodes=1 --partition=medium --mem=16GB --gres=gpu:1 --pty bash -i
# jupyter notebook --ip $(ip addr show eth0 | grep 'inet ' | awk '{print $2}' | cut -d/ -f1) --no-browser
