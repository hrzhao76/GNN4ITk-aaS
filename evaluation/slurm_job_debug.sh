#!/bin/bash
#SBATCH -A m3443
#SBATCH -J GNN4ITk_aaS_perf_eval_1gpu
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --image=hrzhao076/gnn4itk-aas:v0.2
#SBATCH --volume="/global/homes/h/hrzhao/Projects/GNN4ITk-aaS/:/workspace/"
#SBATCH --output=/global/homes/h/hrzhao/Projects/GNN4ITk-aaS/evaluation/slurm/slurm-%j/%j.out
#SBATCH --error=/global/homes/h/hrzhao/Projects/GNN4ITk-aaS/evaluation/slurm/slurm-%j/%j.err

srun shifter /global/homes/h/hrzhao/Projects/GNN4ITk-aaS/evaluation/run_server_n_perf_analyzer.sh /global/homes/h/hrzhao/Projects/GNN4ITk-aaS/evaluation/slurm/slurm-$SLURM_JOB_ID/
