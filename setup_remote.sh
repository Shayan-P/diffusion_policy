#! /bin/bash

# install miniconda

if ! command -v conda &> /dev/null; then
    mkdir -p ~/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm ~/miniconda3/miniconda.sh

    ~/miniconda3/bin/conda init

    source ~/.bashrc
fi

# create the conda environment

conda env create -f conda_environment.yaml
conda activate robodiff

# override the current installed version of huggingface-hub
pip install huggingface-hub==0.25.0

if ! command -v unzip &> /dev/null; then
    apt install -y unzip
fi

# get the data
mkdir data && cd data
wget https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip
unzip pusht.zip && rm -f pusht.zip && cd ..
