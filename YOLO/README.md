# YOLO learning

## Windows Installation

In PyCharm IDE, create a virtual environment as follows:

- Conda with python version 3.12
- Name: YOLO11
- Path to Conda: C:\miniconda3\condabin\conda.bat

Open Terminal in PyCharm and install PyTorch
```
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

Then install YOLO
```
pip install ultralytics
```

> In order to use PyTorch with an NVIDIA GPU, all we need to do is install PyTorch binaries and start using it as PyTorch is shipped with everything in-built
> (PyTorch binaries include CUDA, CuDNN, NCCL, MKL, etc.)
