# Use NVIDIA Triton Server as the base image
FROM nvcr.io/nvidia/tritonserver:24.05-pyt-python-py3

LABEL description="Acorn dependencies based on the Triton Inference Server Python backend." \
      maintainer="Haoran Zhao <haoran.zhao@cern.ch>" \
      version="0.2"

ENV DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_ARCH_LIST="8.0+PTX" \
    CUDA_ARCH="80;86"
    
# Set the working directory to /opt
WORKDIR /opt

# Install necessary packages, set up python alias and clean up in one layer
RUN apt-get update && apt-get install -y \
    cmake \
    uuid-dev \
    openssl \
    libssl-dev \
    rapidjson-dev \
    libb64-dev \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Clone the repository into the /opt/acorn directory
RUN git clone https://gitlab.cern.ch/gnn4itkteam/acorn.git

# Change the working directory to /opt/acorn
WORKDIR /opt/acorn

# Change the branch 
RUN git checkout dmurnane_cicd_pipeline_test

# Install Python dependencies and the acorn package in one step
RUN pip install torch==2.1.0+cu121 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121 \
    && pip install wandb \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install -e . 

# No FRNN or prefix atm
# RUN pip install git+https://github.com/xju2/FRNN.git \
#     && pip install git+https://github.com/asnaylor/prefix_sum.git \
#     && rm -rf /root/.cache/pip

# Install triton client
RUN cd /tmp/ && git clone https://github.com/triton-inference-server/client.git \
    && cd client && mkdir build && cd build \
    && cmake .. -DTRITON_ENABLE_CC_HTTP=OFF \
    -DTRITON_ENABLE_CC_GRPC=ON \
    -DTRITON_ENABLE_GPU=ON \
    -DTRITON_ENABLE_PERF_ANALYZER=ON \
    -DTRITON_ENABLE_PERF_ANALYZER_C_API=ON \
    -DCMAKE_CXX_STANDARD=17 \
    -DTRITON_COMMON_REPO_TAG=r24.05 \
    -DTRITON_THIRD_PARTY_REPO_TAG=r24.05 \
    -DTRITON_CORE_REPO_TAG=r24.05 \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    && make -j20 cc-clients \
    && cd ../.. && rm -rf client

RUN apt-get clean -y && rm -rf /var/lib/apt/lists/* && rm -rf /tmp/*

WORKDIR /workspace/

# ENTRYPOINT ["python", "/opt/acorn/check_acorn.py"]
# CMD ["/bin/bash"]