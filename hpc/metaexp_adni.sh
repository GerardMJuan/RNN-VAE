#!/bin/bash
#SBATCH -J mcrvae
#SBATCH -p medium
#SBATCH --mem 16G
#SBATCH --gres=gpu:1
#SBATCH -o LOGS/mcrvae_revision.out # STDOUT
#SBATCH -e LOGS/mcrvae_revision.err # STDERR

export PATH="$HOME/project/anaconda3/bin:$PATH"
source activate dlnn38

module --ignore-cache load CUDA/11.4.3
module --ignore-cache load cuDNN/8.2.4.15-CUDA-11.4

# python scripts_small/metaexp_adni_full.py
nvidia-smi
python scripts_extra_exp/evaluate_sensitivity.py