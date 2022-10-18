#!/bin/bash
#SBATCH -J vae_bl
#SBATCH -p high
#SBATCH --workdir=/homedtic/gmarti/CODE/RNN-VAE
#SBATCH --mem 32G
#SBATCH --gres=gpu:1
#SBATCH -o LOGS/high_%J.out # STDOUT
#SBATCH -e LOGS/high_%j.err # STDERR

source /etc/profile.d/lmod.sh
source /etc/profile.d/easybuild.sh
# source activate dlnn38

#source activate /homedtic/gmarti/ENV/dl38
module load Python
module --ignore-cache load CUDA/10.2.89
module --ignore-cache load cuDNN/7.6.5.32-CUDA-10.2.89
source /homedtic/gmarti/pytorch/bin/activate 

# python scripts_mc/metaexp_adni.py
python scripts_mc/metaexp_adni_bl.py