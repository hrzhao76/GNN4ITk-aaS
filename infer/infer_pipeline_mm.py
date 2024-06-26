import pandas as pd 
import torch 

from acorn.stages.graph_construction.graph_construction_stage import EventDataset
from acorn.stages.graph_construction.models.py_module_map import PyModuleMap

from acorn.stages.edge_classifier.edge_classifier_stage import GraphDataset
from acorn.stages.edge_classifier import InteractionGNN2
from torch_geometric.data import Data, Batch

from acorn.stages.track_building import ConnectedComponents
from acorn.stages.track_building.utils import load_reconstruction_df

torch.set_float32_matmul_precision("highest")
workdir = "/global/cfs/cdirs/m3443/data/GNN4ITk-aaS/dev_mm"
data_name = "trainset" # select one event from {data_name}
event_name = "event005000001"

hparams_mm = {
    "accelerator" : "gpu",
    "module_map_path": f"{workdir}/models/MMtriplet_1GeV_3hits_noE__merged__sorted.txt",
    "input_dir" : f"{workdir}/data/",
    "stage_dir" : "./stages/module_map/",
    # "log_level": "info",
    "data_split":[1, 0, 0],
    "batch_size": 2000000, # inside one event, might cause memory issue
}

event_mm = EventDataset(
    input_dir = hparams_mm['input_dir'],
    data_name = data_name,
    num_events = 1,
    use_csv = True,
    hparams=hparams_mm)

module_map = PyModuleMap(hparams_mm)
print("[Module Map]: building graphs...")
# module_map.build_graphs(event_mm, event_mm.data_name)

# essentiall call build_graph() which return graph 
# could change this to avoid file I/O 
# graph, particles, hits = event_mm.get(0)
graph_path = f"{workdir}/data/trainset/{event_name}-graph.pyg"
csv_truth_path = f"{workdir}/data/trainset/{event_name}-truth.csv"
graph = torch.load(graph_path)
graph = event_mm.preprocess_graph(graph)
hits = pd.read_csv(csv_truth_path)
graph = module_map.build_graph(graph, hits)

hparams_gnn = {
    "accelerator" : "gpu",
    "checkpoint": "./models/GNN_IN2_epochs169.ckpt",
    "input_dir" : "./stages/module_map",
    "stage_dir" : "./stages/gnn/",
    "log_level": "info",
    "data_split":[1, 0, 0],
    "undirected": False,
}
gnn_model = InteractionGNN2.load_from_checkpoint(
    hparams_gnn['checkpoint'], 
    map_location="cuda:0")
hparams_gnn = {**gnn_model._hparams, **hparams_gnn}
graph_dataset = GraphDataset(
    input_dir= hparams_gnn['input_dir'],
    data_name = data_name,
    num_events = 0,
    preprocess = True,
    hparams= hparams_gnn,
)
event = graph_dataset.preprocess_event(graph)

# event = graph_dataset.get(idx = 0)
# event = event.to('cuda:0')

# TODO: check the processing with the module_map.build_graph()
# Could avoid file I/O? 

# unnecessary, just to match the format w/ DataBatch
# event = Batch.from_data_list([event]) 

gnn_model._hparams = {**gnn_model._hparams, **hparams_gnn}

print("[GNN]: Forwarding...")
with torch.no_grad():
    output = gnn_model.forward(event)
scores = torch.sigmoid(output)
event.scores = scores.detach()

# gnn_model.save_edge_scores(event, graph_dataset)

# hparams_trk_cc = {
#     "stage_dir" : "./stages/track/"
# }
track_builder_cc = ConnectedComponents({})
event.to('cpu')
print("[Track building CC:]: building...")

graph = track_builder_cc._build_event(event)
d = load_reconstruction_df(graph)
# include distance from origin to sort hits
d["r2"] = (graph.r**2 + graph.z**2).cpu().numpy()
# Keep only hit_id associtated to a tracks (label >= 0, not -1), sort by track_id and r2
d = d[d.track_id >= 0].sort_values(["track_id", "r2"])
# Make a dataframe of list of hits (one row = one list of hits, ie one track)
tracks = d.groupby("track_id")["hit_id"].apply(list)

# track_builder_cc._build_and_save(event, hparams_trk_cc["stage_dir"])


# essentially call _build_event() and apply a filter 
# could change this to avoid file I/O 

print("DONE!")

