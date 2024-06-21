####################################################
# This script prepares the input for the GNN4ITk   #
# Module Map approach by converting the CSV files  #
# Author: Haoran Zhao                              #
# Email: haoran.zhao [at] cern.ch                  #
# Date: Jan 2024                                   #
####################################################
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import os, sys

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

def process_csv_and_convert(csv_path: Path, output_json_path: Path = None):
    if output_json_path is None:
        if csv_path.suffix == ".csv":
            output_json_path = csv_path.parent / f"{csv_path.stem}.json"
    if csv_path.is_dir():
        output_json_path = csv_path / f"{csv_path.stem}.json"

    csv_content_list = []
    # Read the CSV file into a pandas DataFrame
    if csv_path.suffix == ".csv" and csv_path.is_file():
        name, _ = csv_path.split("-")
        particles_path = f"{name}-particles.csv"
        truth_path = f"{name}-truth.csv"
        hits, particles = load_data(particles_path, truth_path)
        csv_content_list.append([hits, particles])
    elif csv_path.is_dir():
        print(f"Reading all CSV files in {csv_path}")
        particles_files_path = sorted(csv_path.glob("event*-particles.csv"))
        truth_files_path = sorted(csv_path.glob("event*-truth.csv"))
        for particles_path, truth_path in zip(particles_files_path, truth_files_path):
            hits, particles = load_data(particles_path, truth_path)
            csv_content_list.append([hits, particles])
    else:
        raise NotImplementedError(f"Unsupported file type: {csv_path.suffix}")

    # Convert the DataFrame to a flattened list
    json_content = convert_json(csv_content_list)
    with open(output_json_path, "w") as json_file:
        json.dump(json_content, json_file, indent=4)

input_features_dict = {
    "FEATURES_HITS": "FP64",
    "FEATURES_PARTICLES": "FP64",
}

hit_hardware_dict = {
    "PIXEL" : 0,
    "STRIP" : 1
}

def convert_json(csv_df_list: list):
    json_format_list = []

    for hits, particles in csv_df_list:
        hits["hardware"] = hits["hardware"].apply(lambda x: hit_hardware_dict[x])
        json_format_list.append(
            {
                "FEATURES_HITS": {
                    "content": hits.values.flatten().tolist(),
                    "shape": list(hits.shape),
                },
                "FEATURES_PARTICLES": {
                    "content": particles.values.flatten().tolist(),
                    "shape": list(particles.shape),
                },
            }
        )
    return {"data": json_format_list}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", type=str, required=True)
    parser.add_argument("--output-json-path", type=str, required=False, default=None)
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    output_json_path = (
        None if args.output_json_path is None else Path(args.output_json_path)
    )
    process_csv_and_convert(csv_path, output_json_path)