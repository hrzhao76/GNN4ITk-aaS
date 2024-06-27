# Evaluation
## Submit slurm jobs

``` bash
sbatch slurm_submit.sh
```

One example the ouptut of `perf_analyzer` is ['./ex_perf_output.csv'](./ex_perf_output.csv).

# Development Notes
## Generate json file for perf_analyzer

``` bash 
python gen_json.py --csv-path /global/cfs/cdirs/m3443/data/GNN4ITk-aaS/dev_mm/data/testset/ 

# the output file is 
# /global/cfs/cdirs/m3443/data/GNN4ITk-aaS/dev_mm/data/testset/testset.json
```

``` bash 
export log_filename=test_server.log
nohup tritonserver --model-repository=/workspace/backend/ --log-verbose=4  > ${log_filename} 2>&1 &
server_pid=$!

perf_analyzer -m GNN4ITk_MM_Infer --percentile=95 -i grpc --input-data /global/cfs/cdirs/m3443/data/GNN4ITk-aaS/dev_mm/data/testset/testset.json --measurement-mode count_windows --measurement-request-count 10 -f perf_output.csv --verbose-csv --collect-metrics
```


## Request a node for interactive session
``` bash 
srun -C gpu -q interactive -N 1 -G 1 -c 32 -t 4:00:00 -A m3443 --image=hrzhao076/gnn4itk-aas:v0.2 --volume="/global/homes/h/hrzhao/Projects/GNN4ITk-aaS/:/workspace/" --pty /bin/bash -l
```