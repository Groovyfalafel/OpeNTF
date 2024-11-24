# Use my custom cuda_ubuntu image
FROM kmthang/cuda:11.1-ubuntu20.04

# Add label to the image and version to the image
LABEL maintainer="thangk@uwindsor.ca"
LABEL version="1.0"

RUN apt-get update && apt-get install -y \
    python3 \
    python3-dev \
    python3-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*


RUN pip install --upgrade pip setuptools

# Install Python and CUDA-related packages directly using pip
RUN pip install --root-user-action=ignore \
        numpy \
        pandas \
        scikit-learn \
        scipy \
        matplotlib \
        gensim \
        tqdm \
        PyYAML \
        pytrec-eval-terrier \
        python-dateutil

# Install torch, torchvision, and torchaudio
RUN pip install --root-user-action=ignore \
    torch==1.9.0+cu111 torchvision torchaudio \
    torch-scatter==2.0.9 torch-sparse==0.6.12 torch-cluster==1.5.9 torch-spline-conv==1.2.1 \
    torch-geometric==2.1.0 \
    bayesian-torch \
    -f https://download.pytorch.org/whl/cu111/torch_stable.html \
    -f https://data.pyg.org/whl/torch-1.9.0+cu111.html


# Set the working directory
WORKDIR /OpeNTF

# Set entrypoint
ENTRYPOINT ["/bin/bash"]
