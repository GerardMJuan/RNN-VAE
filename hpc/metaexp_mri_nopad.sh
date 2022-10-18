#!/bin/bash
#SBATCH -J rnnvae
#SBATCH -p high
#SBATCH --workdir=/homedtic/gmarti/CODE/RNN-VAE
#SBATCH --mem 16G
#SBATCH --gres=gpu:1
#SBATCH -o LOGS/vae1_%J.out # STDOUT
#SBATCH -e LOGS/vae1_%j.err # STDERR

source /etc/profile.d/lmod.sh
source /etc/profile.d/easybuild.sh
# source activate dlnn38

#source activate /homedtic/gmarti/ENV/dl38
module load Python
module --ignore-cache load CUDA/10.2.89
module --ignore-cache load cuDNN/7.6.5.32-CUDA-10.2.89
source /homedtic/gmarti/pytorch/bin/activate 

python scripts/metaexp_mri_nopad.py