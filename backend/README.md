# Triton Server
## Module Map Inference 
Currently, only the module map is deployed. The Python backend, `model.py`, depends on the class `GNNMMInferencePipeline`, which is factored out into [`module_map_pipeline.py`](../standalone/module_map_pipeline.py). A significant amount of `acorn` code is utilized in the class due to the preprocessing required at each stage.  

The backend operates by receiving a string representing the event ID from the client, indicating which event should be loaded. Subsequently, the server runs the inference pipeline. This process is necessary due to the large size of the input data, approximately 100 MB per event.

### Server 
``` bash
# pull the docker image with acorn dependencies 
podman-hpc pull hrzhao076/gnn4itk-aas:v0.1

# start the server 
podman-hpc run -it --gpu --rm --shm-size=2g -p8000:8000 -p8001:8001 -p8002:8002 -v ${PWD}:/workspace/ -v /global/cfs/cdirs/m3443/data/GNN4ITk-aaS/dev_mm/:/global/cfs/cdirs/m3443/data/GNN4ITk-aaS/dev_mm/ hrzhao076/gnn4itk-aas:v0.1 /bin/bash

# inside the container
tritonserver --model-repository=/workspace/backend/
```

### Client
``` bash 
# client 
podman-hpc run -it --rm --net=host -v ${PWD}:/workspace/ -v /global/cfs/cdirs/m3443/data/GNN4ITk-aaS/dev_mm/:/global/cfs/cdirs/m3443/data/GNN4ITk-aaS/dev_mm/ nvcr.io/nvidia/tritonserver:24.04-py3-sdk /bin/bash

python /workspace/backend/GNN4ITk_MM_Infer/client.py

```
Results should be something like this:  
```
HIT_IDs: [     2   1041     86 ... 276900 272239 275962]
TRACK_IDs: [   1    1    1 ... 2124 2125 2125]
```

# Dev Notes 
- [ ] Wrapper in the client to make 2D jagged array
- [ ] Docker container `prefix_sum` and `FRNN`  
- [ ] Metric learning gnn 