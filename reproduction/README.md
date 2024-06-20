# Reproduction 

This folder utilizes the classic ACORN style to perform analyses on three distinct events, demonstrating outputs identical to those of the standalone version. It features an abstracted Python class, `[module_map_pipeline.py](module_map_pipeline.py)`, which encapsulates the essential inference code, bypassing any intermediate stage files I/O to the disk.    

``` bash 
cp -r /../acorn/examples/CTD_2023/*.yaml ./CTD2023
# Modify the *_infer.yaml and batch_size
data_dir=/global/cfs/cdirs/m3443/data/GNN4ITk-aaS/dev_mm/data/testset/

acorn infer module_map_infer.yaml 

# change infer_stage.py 
# stage_module = stage_module.load_from_checkpoint(checkpoint_path, map_location="cuda:0")
acorn infer gnn_infer.yaml -c /global/cfs/cdirs/m3443/data/GNN4ITk-aaS/dev_mm/models/GNN_IN2_epochs169.ckpt 

# change max_worker to 1. no idea why it is stuck with 8, multi-processing 
# and default setting of CC is applied. 
acorn infer track_building_infer.yaml 

# haven't make track eval working KeyError: 'track_id'
# spacepoint_matching = spacepoint_matching.merge(...)
# acorn eval track_building_eval.yaml

head /global/cfs/cdirs/m3443/data/GNN4ITk-aaS/dev_mm/stages_files/track/testset_tracks/event005000901.txt 

```

and the output is as following, which is exactly identical to the standalone output, [event005000901_reco_trks.txt](../standalone/event005000901_reco_trks.txt).   


``` bash 
856 4
101589 15921 11 15995 68 133 185 48877 17197 17261 17324 56774 52122 60625 60039 60081 79718 92078 92095 243310
101593 15937 15
101587 866 18 929 78 18636 17096 18700 17160 18760 17226 52253 80084 92282 240153 243270 246988
101607 27 81 17098 17161 17225 56752 52109 79675 79709 92052 237591 240219 243336 247115
15973 45 16051 16135 16204 48943 49023 49104 60024 59522 59563 79518 91830 91855 247161 247159
900 50 112 166 221 17238 17298 17359 17424 17490 60676 60730 60780 80139 79951 80005 92345 92235
15983 53 16060 118 16144 48873 48953 49033 49114
904 54 957 115 173 224 18773 17242 18844 17303
93 999
```