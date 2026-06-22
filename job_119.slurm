#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --mem=80GB
#SBATCH --nodes=1
#SBATCH --job-name=job_119
#SBATCH --output=/cluster/home/kristiac/NeCT/run_log_119.txt
#SBATCH --time=30:00:00
#SBATCH --export=ALL
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint="a100|gpu80g"
#SBATCH --ntasks-per-node=4
echo "We are running from this directory: $SLURM_SUBMIT_DIR"
echo "The name of the job is: $SLURM_JOB_NAME"
echo "The job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "Total of $SLURM_NTASKS cores"
echo "Total of GPUS: $CUDA_VISIBLE_DEVICES"
echo "synthetic_slab degree=24 ac=3"
source nect_a100/bin/activate
python -m demo.synthetic_slab --degree 24 --ac 3
