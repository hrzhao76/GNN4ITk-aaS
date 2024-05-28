import torch 
import pandas as pd 
import numpy as np

from acorn.stages.graph_construction.graph_construction_stage import EventDataset
from acorn.stages.graph_construction.models.py_module_map import PyModuleMap

from acorn.stages.edge_classifier.edge_classifier_stage import GraphDataset
from acorn.stages.edge_classifier import InteractionGNN2
from torch_geometric.data import Data, Batch

from acorn.stages.track_building import ConnectedComponents
from acorn.stages.track_building.utils import load_reconstruction_df

class GNNMMInferencePipeline():
    def __init__(self,
                 workdir: str = "/global/cfs/cdirs/m3443/data/GNN4ITk-aaS/dev_mm",
                 data_name: str = "trainset",
                 mm_batch_size: int = 500000) -> None:
        self.workdir = workdir
        self.data_name = data_name # select one event from {data_name}

        torch.set_float32_matmul_precision("highest")
        torch.set_grad_enabled(False)
        hparams_mm = {
            "accelerator" : "gpu",
            "module_map_path": f"{workdir}/models/MMtriplet_1GeV_3hits_noE__merged__sorted.txt",
            "input_dir" : f"{workdir}/data/",
            "stage_dir" : "./stages/module_map/",
            # "log_level": "info",
            "data_split":[1, 0, 0],
            "batch_size": mm_batch_size, # inside one event, might cause memory issue
        }

        self.event_mm = EventDataset(
            input_dir = hparams_mm['input_dir'],
            data_name = data_name,
            num_events = 1,
            use_csv = True,
            hparams=hparams_mm)
        
        self.module_map = PyModuleMap(hparams_mm)

        hparams_gnn = {
            "accelerator" : "gpu",
            "checkpoint": f"{workdir}/models/GNN_IN2_epochs169.ckpt",
            "input_dir" : "./stages/module_map",
            "stage_dir" : "./stages/gnn/",
            "log_level": "info",
            "data_split":[1, 0, 0],
            "undirected": False,
        }
        self.gnn_model = InteractionGNN2.load_from_checkpoint(
            hparams_gnn['checkpoint'], 
            map_location="cuda:0")
        # TODO: Check this 
        # otherwise KeyError: 'dr' 
        hparams_gnn = {**self.gnn_model._hparams, **hparams_gnn}
        self.gnn_model._hparams = {**self.gnn_model._hparams, **hparams_gnn}
    
        self.graph_dataset = GraphDataset(
            input_dir= hparams_gnn['input_dir'],
            data_name = data_name,
            num_events = 0,
            preprocess = True,
            hparams= hparams_gnn,
        )

        self.track_builder_cc = ConnectedComponents({})

    def load_data(self, graph_path, csv_truth_path):
        graph = torch.load(graph_path)
        hits = pd.read_csv(csv_truth_path)
        return graph, hits 


    def forward(self, graph, hits):
        # graph construction
        print("[Module Map]: building graphs...")
        graph = self.event_mm.preprocess_graph(graph)
        graph = self.module_map.build_graph(graph, hits)

        print("[GNN]: Forwarding...")
        # edge classifier 
        event = self.graph_dataset.preprocess_event(graph)
        output = self.gnn_model.forward(event)

        scores = torch.sigmoid(output)
        event.scores = scores.detach()

        event.to('cpu')
        print("[Track building CC:]: building...")
        graph = self.track_builder_cc._build_event(event)
        d = load_reconstruction_df(graph)
        # include distance from origin to sort hits
        d["r2"] = (graph.r**2 + graph.z**2).cpu().numpy()
        # Keep only hit_id associtated to a tracks (label >= 0, not -1), sort by track_id and r2
        d = d[d.track_id >= 0].sort_values(["track_id", "r2"])
        # Make a dataframe of list of hits (one row = one list of hits, ie one track)
        tracks = d.groupby("track_id")["hit_id"].apply(list)

        return tracks
    
    def infer(self, graph_path, csv_truth_path):
        graph, hits  = self.load_data(graph_path, csv_truth_path)
        return self.forward(graph, hits)

if __name__ == '__main__':
    event_name = 'event005000001'
    data_name ='trainset'
    pipeline = GNNMMInferencePipeline(mm_batch_size=1000000)
    input_folder = f"{pipeline.workdir}/data/{data_name}/"
    graph_path = input_folder + f"{event_name}-graph.pyg"
    csv_truth_path = input_folder + f"{event_name}-truth.csv"
    pipeline.infer(graph_path, csv_truth_path)