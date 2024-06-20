# GNN4ITk-aaS

## Setup 
``` bash 
git submodule add ssh://git@gitlab.cern.ch:7999/gnn4itkteam/acorn.git acorn && cd acorn && git submodule update --init --recursive
git checkout dev_AL_inference

mamba create --name acorn python=3.10 && mamba activate acorn
pip install torch==2.1.0 && pip install --no-cache-dir -r requirements.txt
pip install wandb
pip install -e . 


pip install git+https://github.com/asnaylor/prefix_sum.git
pip install git+https://github.com/xju2/FRNN.git

```

``` bash
export project_dir=/global/homes/h/hrzhao/Projects/GNN4ITk-aaS
export data_dir=/global/cfs/cdirs/m3443/data/GNN4ITk-aaS/CTD2023

# a small dataset 
# /global/cfs/cdirs/m3443/data/GNN4ITk-aaS/dev_mm/data/testset/
```

