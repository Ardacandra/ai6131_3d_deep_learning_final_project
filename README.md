# AI6131-3D Deep Learning Final Project
This repository contains my work for the final project of NTU MSAI AI6131-3D Deep Learning class.

### Setup Instructions

1. Clone the repository

```bash
git clone https://github.com/Ardacandra/ai6131_3d_deep_learning_final_project.git
cd ai6131_3d_deep_learning_final_project
```

2. Prepare conda environment

```bash
# Create the environment
conda create -n ai6131_3d_deep_learning_final_project python=3.10 -y
conda activate ai6131_3d_deep_learning_final_project

# Install Core DL Stack (PyTorch + CUDA)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install project dependencies
pip install -r requirements.txt
```