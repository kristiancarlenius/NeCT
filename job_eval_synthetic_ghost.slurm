#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --mem=60GB
#SBATCH --nodes=1
#SBATCH --job-name=eval_synthetic_ghost
#SBATCH --output=/cluster/home/kristiac/NeCT/run_log_eval_synthetic_ghost.txt
#SBATCH --time=01:00:00
#SBATCH --export=ALL
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint="a100|gpu80g"
#SBATCH --ntasks-per-node=1
echo "We are running from this directory: $SLURM_SUBMIT_DIR"
echo "The name of the job is: $SLURM_JOB_NAME"
echo "The job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Total of GPUS: $CUDA_VISIBLE_DEVICES"
source nect_a100/bin/activate
python -m demo.eval_synthetic_ghost --outputs-dir /cluster/home/kristiac/NeCT/outputs  --gt-dir /cluster/home/kristiac/NeCT/Datasets/synthetic_ghost --out-dir /cluster/home/kristiac/NeCT/results/synthetic_ghost_eval --gif --gif-frames 100
