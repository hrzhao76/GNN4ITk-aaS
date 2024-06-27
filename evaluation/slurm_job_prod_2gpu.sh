#!/bin/bash
#SBATCH -A m3443
#SBATCH -J GNN4ITk_aaS_perf_eval_2gpu
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 4:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 32
#SBATCH --gpus-per-task=2
#SBATCH --image=hrzhao076/gnn4itk-aas:v0.2
#SBATCH --volume="/global/homes/h/hrzhao/Projects/GNN4ITk-aaS/:/workspace/"
#SBATCH --output=/global/homes/h/hrzhao/Projects/GNN4ITk-aaS/evaluation/slurm/slurm-%j/%j.out
#SBATCH --error=/global/homes/h/hrzhao/Projects/GNN4ITk-aaS/evaluation/slurm/slurm-%j/%j.err

srun shifter /global/homes/h/hrzhao/Projects/GNN4ITk-aaS/evaluation/run_server_n_perf_analyzer.sh --output-path /global/homes/h/hrzhao/Projects/GNN4ITk-aaS/evaluation/slurm/slurm-$SLURM_JOB_ID/ --concurrency-range "1:4:1"
