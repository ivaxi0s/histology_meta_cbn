#!/bin/bash
#SBATCH --mail-user=ivaxi-miteshkumar.sheth.1@ens.etsmtl.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --job-name=try
#SBATCH --output=%x-%j.out
#SBATCH --nodes=32
#SBATCH --gres=gpu:1
#SBATCH --mem=100GB
#SBATCH --time=2:00:00
#SBATCH --account=def-ebrahimi
​#wandb login b0ebb8272d653169adb078a4e3f70cb1ebbf41c0 

source ~/envs/cbn/bin/activate
wandb offline
python train.py -data_path='/home/ivsh/scratch/datasets/pcam'