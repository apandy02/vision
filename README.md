# Computer Vision

In this repository, I implement computer vision techniques and ideas that are initially new to me in notebooks. 
Then, if the notebook contains blocks of code that can be re-used for different vision tasks, I put them in files 
under the `src` directory. 

In this iterative fashion, I'm building a mini vision library built off of pure PyTorch. 

This repository goes from Lenet-5, to diffusion. 

## Directory Structure

```
vision/
├── nbs/                    # Jupyter notebooks with implementations
│   ├── generative/         # Generative models organized by type
│   │   ├── autoencoders/   # Autoencoder implementations
│   │   ├── diffusion/      # Diffusion model implementations
│   │   └── gans/           # Generative Adversarial Networks
│   ├── video/              # Video understanding and processing
│   ├── augmentation/       # Traffic sign recognition project (GTSRB)
│   └── [other notebooks]   # Classification, segmentation, etc.
├── src/                    # Reusable PyTorch modules and utilities
│   └── vision/             # Core vision library modules
│       └── transformers/   # Transformer-related modules
├── pyproject.toml          # Project configuration
└── README.md              # This file
```

## Notebooks (`nbs/`)

### Core Vision Tasks
- `mnist_lenet.ipynb`: training LeNet-5 on MNIST
- `resnets.ipynb`: residual networks
- `fcn_segmentation.ipynb`: fully convolutional networks for semantic segmentation
- `mixture_of_experts.ipynb`: implementing a sparsely-gated mixture of experts with ResNets
- `vit.ipynb`: vision transformer implementation

### Video Understanding (`nbs/video/`)
- `timesformer.ipynb`: implementing TimeSFormer for video classification

### Data Augmentation (`nbs/augmentation/`)
- `GTSRBDataAugmentation.ipynb`: data augmentation techniques exploration
- `Project2GTSRB.ipynb`: transfer learning project using the German Traffic Sign Recognition Benchmark (GTSRB) dataset with ResNet architecture and extensive data augmentation techniques

### Generative Models (`nbs/generative/`)

#### Autoencoders (`autoencoders/`)
- `autoencoders.ipynb`: autoencoder implementations

#### Diffusion Models (`diffusion/`)
- `ddpm_mnist.ipynb`: implementing diffusion models architecture on MNIST
- `ddpm_cifar.ipynb`: applying the DDPM architecture from `src` to CIFAR-10
- `ddim.ipynb`: implementing denoising diffusion implicit models

#### Generative Adversarial Networks (`gans/`)
- `dcgan.ipynb`: Deep Convolutional Generative Adversarial Networks

## Source Code (`src/`)

Polished, reusable modules extracted from notebooks:

### Core Vision Library (`src/vision/`)
- `autoencoders.py`: autoencoder and variational autoencoder implementations
- `resnet.py`: building blocks for residual networks
- `unet.py`: building blocks for UNet for image generation
- `ddpm.py`: training and sampling for denoising diffusion probabilistic models
- `utils.py`: utility functions

### Transformer Modules (`src/vision/transformers/`)
- `attention.py`: attention mechanisms and building blocks
- `blocks.py`: transformer block implementations


`Post GenAI Era disclaimer`: I have tried to re-implement papers from scratch here, and have cursor tab disabled while developing things. I have used LLMs to help me understand things, find bugs, optimize training and so on, but the main goal here was to learn things, so the 
first pass of each implementation (the foundation) was written by a human :)