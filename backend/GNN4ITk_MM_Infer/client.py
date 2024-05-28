import tritonclient.grpc as grpcclient
import numpy as np

def main():
    # Create Triton client
    url = "localhost:8001"  # Replace with your Triton server address
    model_name = "GNN4ITk_MM_Infer"
    triton_client = grpcclient.InferenceServerClient(url=url)

    # Set up input
    event_name = "event005000001"
    inputs = []
    inputs.append(grpcclient.InferInput("EVENT_ID", [1], "BYTES"))

    # Convert input data to bytes
    input_data = np.array([event_name], dtype=np.object_)
    inputs[0].set_data_from_numpy(input_data)

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
