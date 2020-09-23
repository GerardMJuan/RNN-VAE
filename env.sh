#!/bin/bash
source /etc/profile.d/lmod.sh
source /etc/profile.d/easybuild.sh
export PATH="$HOME/project/anaconda3/bin:$PATH"
source activate dlnn2
module load CUDA/10.2.89
module load cuDNN/7.6.5.32-CUDA-10.2.89 
export XDG_RUNTIME_DIR=""
#srun --nodes=1 --partition=medium --mem=16GB --gres=gpu:1 --pty bash -i
