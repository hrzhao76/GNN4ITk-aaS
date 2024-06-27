#!/bin/bash

for i in {1..4}
do
    sbatch slurm_job_prod_${i}gpu.sh
done