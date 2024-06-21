import tritonclient.grpc as grpcclient
import numpy as np
import pandas as pd
from pathlib import Path

def load_data(particles_path, truth_path):
    particles = pd.read_csv(particles_path)
    hits = pd.read_csv(truth_path)
    hit_features = [
        'hardware', 'barrel_endcap', 'layer_disk', 
        'eta_module', 'phi_module', 'module_id', 'region',
        'hit_id', 'x', 'y', 'z', 'particle_id', 
        'cluster_index_1', 'cluster_x_1', 'cluster_y_1', 'cluster_z_1',
        'cluster_index_2', 'cluster_x_2', 'cluster_y_2', 'cluster_z_2', 
    ]
    particle_features = [
        'particle_id', 
        'subevent', 'barcode', 
        'px', 'py', 'pz', 'pt', 'eta',
        'vx', 'vy', 'vz', 
        'radius', 'status', 'charge', 
        'pdgId'
    ]

    hits = hits[hit_features]
    particles = particles[particle_features]
    return hits, particles


def main():
    # Create Triton client
    url = "localhost:8001"  # Replace with your Triton server address
    model_name = "GNN4ITk_MM_Infer"
    triton_client = grpcclient.InferenceServerClient(url=url)

    # Load data from CSV files
    workdir = Path("/global/cfs/cdirs/m3443/data/GNN4ITk-aaS/dev_mm/data")
    data_name = "trainset"
    event_name = "event005000001"
    particles_path = workdir / data_name / f"{event_name}-particles.csv" 
    truth_path = workdir / data_name / f"{event_name}-truth.csv" 
    hits, particles = load_data(particles_path, truth_path)

    hit_hardware_dict = {
        "PIXEL" : 0,
        "STRIP" : 1
    }

    hits["hardware"] = hits["hardware"].apply(lambda x: hit_hardware_dict[x])
    # Prepare input data
    inputs = []
    inputs.append(grpcclient.InferInput("FEATURES_HITS", hits.shape, "FP64"))
    inputs[0].set_data_from_numpy(hits.values.astype(np.float64))

    inputs.append(grpcclient.InferInput("FEATURES_PARTICLES", particles.shape, "FP64"))
    inputs[1].set_data_from_numpy(particles.values.astype(np.float64))

    # Set up output
    outputs = []
    outputs.append(grpcclient.InferRequestedOutput("HIT_IDs"))
    outputs.append(grpcclient.InferRequestedOutput("TRACK_IDs"))

    # Perform inference
    response = triton_client.infer(model_name, inputs=inputs, outputs=outputs)

    # Get output data
    HIT_IDs_data = response.as_numpy("HIT_IDs")
    TRACK_IDs_data = response.as_numpy("TRACK_IDs")

    print("HIT_IDs:", HIT_IDs_data)
    print("TRACK_IDs:", TRACK_IDs_data)

if __name__ == "__main__":
    main()
