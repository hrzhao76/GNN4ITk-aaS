import torch 
import pandas as pd 
import numpy as np

from acorn.stages.data_reading.data_reading_stage import EventReader

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
        confif_eventreader = {
            "feature_sets":{
                "hit_features": ["hit_id", "x", "y", "z", "r", "phi", "eta", "region", "module_id",
                            "cluster_x_1", "cluster_y_1", "cluster_z_1", "cluster_x_2", "cluster_y_2", "cluster_z_2",
                            "cluster_r_1", "cluster_phi_1", "cluster_eta_1", "cluster_r_2", "cluster_phi_2", "cluster_eta_2",
                            "norm_x_1", "norm_y_1", "norm_x_2", "norm_y_2", "norm_z_1", 
                            "eta_angle_1", "phi_angle_1", "eta_angle_2", "phi_angle_2", "norm_z_2"],

                "track_features": ["particle_id", "pt", "radius", "primary", "nhits", "pdgId", "eta_particle", "redundant_split_edges"]
            },
            "region_labels":{
                1: {'hardware': 'PIXEL', 'barrel_endcap': -2},
                2: {'hardware': 'STRIP', 'barrel_endcap': -2},
                3: {'hardware': 'PIXEL', 'barrel_endcap': 0},
                4: {'hardware': 'STRIP', 'barrel_endcap': 0},
                5: {'hardware': 'PIXEL', 'barrel_endcap': 2},
                6: {'hardware': 'STRIP', 'barrel_endcap': 2}
            }
        }
        self.reader = EventReader(config=confif_eventreader)

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

    def load_data_csvonly(self, particles_path, truth_path):
        particles = pd.read_csv(particles_path)
        hits = pd.read_csv(truth_path)
        hit_features = [
            'hardware', 'barrel_endcap', 'layer_disk', 
            'eta_module', 'phi_module', 'module_id', 'region',
            'hit_id', 'x', 'y', 'z', 'particle_id', 
            'cluster_index_1', 'cluster_x_1', 'cluster_y_1', 'cluster_z_1',
            # 'norm_x_1', 'norm_y_1', 'norm_z_1', 'particle_id_1',
            'cluster_index_2', 'cluster_x_2', 'cluster_y_2', 'cluster_z_2', 
            # 'norm_x_2', 'norm_y_2', 'norm_z_2', 'particle_id_2',
            ]
        particle_features = [
            'particle_id', 
            'subevent', 'barcode', 
            'px', 'py', 'pz', 'pt', 'eta',
            'vx', 'vy', 'vz', 
            'radius', 'status', 'charge', 
            'pdgId',
        ]

        hits = hits[hit_features]
        particles = particles[particle_features]
        return hits, particles
    
    def convert_pyG(self, hits, particles, event_id = '005000001'):
        particles = particles.rename(columns={"eta": "eta_particle"})
        hits, particles = self.reader._merge_particles_to_hits(hits, particles)
        hits = self.reader._add_handengineered_features(hits)
        hits = self.reader._clean_noise_duplicates(hits)
        # wrong reader.. 
        # fixed at https://gitlab.cern.ch/gnn4itkteam/acorn/-/commit/b20821c336ea63b3aadc03fd018178d50175dc16#9b834450698a186cdf44d4557ac48cf865a1b1f2_314_312
        # not able to run with the dev branch because of MM1 not found 
        for i in [1, 2]:
            hits[f"cluster_r_{i}"] = np.sqrt(
                hits[f"cluster_x_{i}"] ** 2 + hits[f"cluster_y_{i}"] ** 2
                )
            hits[f"cluster_phi_{i}"] = np.arctan2(
                hits[f"cluster_y_{i}"],
                hits[f"cluster_x_{i}"],
                )
            
            hits[f"cluster_eta_{i}"] = self.calc_eta(
                hits[f"cluster_r_{i}"],
                hits[f"cluster_z_{i}"],
                )
        
        tracks, track_features, hits = self.reader._build_true_tracks(hits)
        hits, particles, tracks = self.reader._custom_processing(hits, particles, tracks)
        graph = self.reader._build_graph(hits, tracks, track_features, event_id)

        return graph, hits 

    @staticmethod
    def calc_eta(r, z):
        theta = np.arctan2(r, z)
        return -1.0 * np.log(np.tan(theta / 2.0))
    
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
    
    def infer_csvonly(self, particles_path, csv_truth_path): 
        hits, particles = self.load_data_csvonly(particles_path, csv_truth_path)
        graph, hits  = self.convert_pyG(hits, particles)
        return self.forward(graph, hits)

if __name__ == '__main__':
    event_name = 'event005000001'
    data_name ='trainset'
    pipeline = GNNMMInferencePipeline(mm_batch_size=1000000)
    input_folder = f"{pipeline.workdir}/data/{data_name}/"
    graph_path = input_folder + f"{event_name}-graph.pyg"
    csv_truth_path = input_folder + f"{event_name}-truth.csv"
    particles_path = input_folder + f"{event_name}-particles.csv"
    # tracks = pipeline.infer(graph_path, csv_truth_path)
    
    tracks = pipeline.infer_csvonly(particles_path, csv_truth_path)

    print(f"Length of the tracks: {len(tracks)}")