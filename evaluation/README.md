

##

``` bash 
python gen_json.py --csv-path /global/cfs/cdirs/m3443/data/GNN4ITk-aaS/dev_mm/data/testset/ 
/global/cfs/cdirs/m3443/data/GNN4ITk-aaS/dev_mm/data/testset/testset.json
```


``` bash 
perf_analyzer -m GNN4ITk_MM_Infer --percentile=95 -i grpc --input-data /global/cfs/cdirs/m3443/data/GNN4ITk-aaS/dev_mm/data/testset/testset.json --measurement-interval 100000

```