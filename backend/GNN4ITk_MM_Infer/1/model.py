import triton_python_backend_utils as pb_utils
import json
import pandas as pd 
import numpy as np

import sys
import os

# Add the standalone directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../standalone')))

from module_map_pipeline import GNNMMInferencePipeline

class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """
    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device
            ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        print('Initialized...')

        self.model_config = json.loads(args["model_config"])
        self.mm_batch_size = int(self.model_config["parameters"]["mm_batch_size"]["string_value"])
        print(f"mm_batch_size: {self.mm_batch_size}")
        self.pipeline = GNNMMInferencePipeline(mm_batch_size=self.mm_batch_size)

        self.hit_hardware_decode = {
            0: "PIXEL",
            1: "STRIP"
        }
        # Extract hits inputs
        self.hit_dtype_dict = {
                'hardware': 'object',
                'barrel_endcap': 'int64',
                'layer_disk': 'int64',
                'eta_module': 'int64',
                'phi_module': 'int64',
                'module_id': 'int64',
                'region': 'float64',
                'hit_id': 'int64',
                'x': 'float64',
                'y': 'float64',
                'z': 'float64',
                'particle_id': 'int64',
                'cluster_index_1': 'int64',
                'cluster_x_1': 'float64',
                'cluster_y_1': 'float64',
                'cluster_z_1': 'float64',
                'cluster_index_2': 'int64',
                'cluster_x_2': 'float64',
                'cluster_y_2': 'float64',
                'cluster_z_2': 'float64'
            }
        self.hit_feature_names = [*self.hit_dtype_dict.keys()]

        # Extract particles inputs
        self.particle_dtype_dict = {
                'particle_id': 'int64',
                'subevent': 'int64',
                'barcode': 'int64',
                'px': 'float64',
                'py': 'float64',
                'pz': 'float64',
                'pt': 'float64',
                'eta': 'float64',
                'vx': 'float64',
                'vy': 'float64',
                'vz': 'float64',
                'radius': 'float64',
                'status': 'int64',
                'charge': 'float64',
                'pdgId': 'int64'
            }
        self.particle_feature_names = [*self.particle_dtype_dict.keys()]

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model.

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        responses = []

        # Every Python backend must iterate through list of requests and create
        # an instance of pb_utils.InferenceResponse class for each of them.
        # Reusing the same pb_utils.InferenceResponse object for multiple
        # requests may result in segmentation faults. You should avoid storing
        # any of the input Tensors in the class attributes as they will be
        # overridden in subsequent inference requests. You can make a copy of
        # the underlying NumPy array and store it if it is required.
        for request in requests:
            try:
                features_hits = pb_utils.get_input_tensor_by_name(request, "FEATURES_HITS")
                features_hits = features_hits.as_numpy()

                features_particles = pb_utils.get_input_tensor_by_name(request, "FEATURES_PARTICLES")
                features_particles = features_particles.as_numpy()
                
                # Create DataFrames from the data
                hits_df = pd.DataFrame(features_hits, columns=self.hit_feature_names)

                hits_df['hardware'] = hits_df['hardware'].apply(lambda x: self.hit_hardware_decode[x])
                particles = pd.DataFrame(features_particles, columns=self.particle_feature_names)

                hits = hits_df.astype(self.hit_dtype_dict)
                particles = particles.astype(self.particle_dtype_dict)
                
                # Assuming you have a self.pipeline object with convert_pyG and forward methods
                graph, hits = self.pipeline.convert_pyG(hits, particles)
                tracks = self.pipeline.forward(graph, hits)

                # Process tracks to create output tensors
                track_lengths = [len(track) for track in tracks]
                hits = np.concatenate(tracks)
                track_ids = np.repeat(np.arange(1, len(tracks) + 1), track_lengths)

                output0 = pb_utils.Tensor("HIT_IDs", hits.astype(np.int32))
                output1 = pb_utils.Tensor("TRACK_IDs", track_ids.astype(np.int32))

                # Create InferenceResponse and append it to responses
                inference_response = pb_utils.InferenceResponse(output_tensors=[output0, output1])
                responses.append(inference_response)
            except Exception as e:
                # If an error occurs, create an InferenceResponse with the error
                inference_response = pb_utils.InferenceResponse(error=pb_utils.TritonError(str(e)))
                responses.append(inference_response)
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')
