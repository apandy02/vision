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
│   └── [other notebooks]   # Classification, segmentation, etc.
├── src/                    # Reusable PyTorch modules and utilities
├── images/                 # Generated images and visualizations
├── augmentation/           # Traffic sign recognition project (GTSRB)
├── pyproject.toml          # Project configuration
└── README.md              # This file
```

## Notebooks (`nbs/`)

### Core Vision Tasks
- `mnist_lenet.ipynb`: training LeNet-5 on MNIST
- `resnets.ipynb`: residual networks
- `fcn_segmentation.ipynb`: fully convolutional networks for semantic segmentation
- `mixture_of_experts.ipynb`: implementing a sparsely-gated mixture of experts with ResNets

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

- `attention2D.py`: all things attention, including 2D attention and its building blocks
- `resnet.py`: building blocks for residual networks
- `unet.py`: building blocks for UNet for image generation
- `ddpm.py`: training and sampling for denoising diffusion probabilistic models
- `utils.py`: utility functions

## Projects

### Traffic Sign Recognition (`augmentation/`)
Transfer learning project using the German Traffic Sign Recognition Benchmark (GTSRB) dataset with ResNet architecture and extensive data augmentation techniques.
