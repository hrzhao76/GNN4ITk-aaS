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

    @staticmethod
    def auto_complete_config(auto_complete_model_config):
        """`auto_complete_config` is called only once when loading the model
        assuming the server was not started with
        `--disable-auto-complete-config`. Implementing this function is
        optional. No implementation of `auto_complete_config` will do nothing.
        This function can be used to set `max_batch_size`, `input` and `output`
        properties of the model using `set_max_batch_size`, `add_input`, and
        `add_output`. These properties will allow Triton to load the model with
        minimal model configuration in absence of a configuration file. This
        function returns the `pb_utils.ModelConfig` object with these
        properties. You can use the `as_dict` function to gain read-only access
        to the `pb_utils.ModelConfig` object. The `pb_utils.ModelConfig` object
        being returned from here will be used as the final configuration for
        the model.

        Note: The Python interpreter used to invoke this function will be
        destroyed upon returning from this function and as a result none of the
        objects created here will be available in the `initialize`, `execute`,
        or `finalize` functions.

        Parameters
        ----------
        auto_complete_model_config : pb_utils.ModelConfig
          An object containing the existing model configuration. You can build
          upon the configuration given by this object when setting the
          properties for this model.

        Returns
        -------
        pb_utils.ModelConfig
          An object containing the auto-completed model configuration
        """
        inputs = [{
            'name': 'EVENT_ID',
            'data_type': 'TYPE_STRING',
            'dims': [1]
        }]
        outputs = [{
            'name': 'HIT_IDs',
            'data_type': 'TYPE_INT32',
            'dims': [-1]
        }, {
            'name': 'TRACK_IDs',
            'data_type': 'TYPE_INT32',
            'dims': [-1]
        }]

        # Demonstrate the usage of `as_dict`, `add_input`, `add_output`,
        # `set_max_batch_size`, and `set_dynamic_batching` functions.
        # Store the model configuration as a dictionary.
        config = auto_complete_model_config.as_dict()
        input_names = []
        output_names = []
        for input in config['input']:
            input_names.append(input['name'])
        for output in config['output']:
            output_names.append(output['name'])

        for input in inputs:
            # The name checking here is only for demonstrating the usage of
            # `as_dict` function. `add_input` will check for conflicts and
            # raise errors if an input with the same name already exists in
            # the configuration but has different data_type or dims property.
            if input['name'] not in input_names:
                auto_complete_model_config.add_input(input)
        for output in outputs:
            # The name checking here is only for demonstrating the usage of
            # `as_dict` function. `add_output` will check for conflicts and
            # raise errors if an output with the same name already exists in
            # the configuration but has different data_type or dims property.
            if output['name'] not in output_names:
                auto_complete_model_config.add_output(output)

        auto_complete_model_config.set_max_batch_size(0)

        # To enable a dynamic batcher with default settings, you can use
        # auto_complete_model_config set_dynamic_batching() function. It is
        # commented in this example because the max_batch_size is zero.
        #
        # auto_complete_model_config.set_dynamic_batching()

        return auto_complete_model_config

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

        self.model_config = model_config = json.loads(args["model_config"])
        self.pipeline = GNNMMInferencePipeline()

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
                # Extract the input tensor from the request
                input0 = pb_utils.get_input_tensor_by_name(request, "EVENT_ID")
                # Convert input tensor to event name
                event_id = input0.as_numpy()[0].decode('utf-8')

                input1 = pb_utils.get_input_tensor_by_name(request, "FEATURE_HITS_HARDWARE")
                feature_hit_hardware = np.char.decode(input1.as_numpy().astype(np.bytes_), 'utf-8')

                input2 = pb_utils.get_input_tensor_by_name(request, "FEATURES_HITS")
                feature_hits = input2.as_numpy()

                input3 = pb_utils.get_input_tensor_by_name(request, "FEATURES_PARTICLES")
                feature_particles = input3.as_numpy()

                # Create DataFrames from the data
                hardware_df = pd.DataFrame(feature_hit_hardware, columns=['hardware'])
                hits_df = pd.DataFrame(feature_hits, columns=self.hit_feature_names[1:])
                hits = pd.concat([hardware_df, hits_df], axis=1)

                particles = pd.DataFrame(feature_particles, columns=self.particle_feature_names)

                hits = hits.astype(self.hit_dtype_dict)
                particles = particles.astype(self.particle_dtype_dict)
                
                # Assuming you have a self.pipeline object with convert_pyG and forward methods
                graph, hits = self.pipeline.convert_pyG(hits, particles, event_id)
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
