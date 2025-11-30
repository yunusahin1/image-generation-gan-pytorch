# Pokemon Image Generation with GAN (PyTorch)

A Generative Adversarial Network (GAN) implementation in PyTorch for generating Pokemon-style images.

## Project Overview

This project implements a Deep Convolutional GAN (DCGAN) to generate new Pokemon images. The GAN learns from a dataset of Pokemon sprites and can generate novel Pokemon-like images.

## Features

- **DCGAN Architecture**: Uses convolutional layers for both Generator and Discriminator
- **Custom Dataset Loader**: Handles Pokemon image dataset with proper preprocessing
- **Training Pipeline**: Complete training loop with loss tracking and checkpointing
- **Image Generation**: Multiple modes for generating images (grid, single, interpolation)
- **Progress Tracking**: Real-time training progress with tqdm
- **Visualization**: Loss plotting and sample image generation during training

## Project Structure

```
image-generation-gan-pytorch/
├── generator.py          # Generator neural network
├── discriminator.py      # Discriminator neural network
├── dataloader.py         # Dataset and DataLoader implementation
├── train.py             # Training script
├── generate.py          # Image generation utilities
├── requirements.txt     # Python dependencies
├── images/              # Pokemon dataset (input images)
├── generated_images/    # Generated samples during training
├── checkpoints/         # Model checkpoints
└── README.md           # This file
```

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the GAN

To train the GAN on the Pokemon dataset:

```bash
python train.py
```

Training configuration can be modified in the `config` dictionary in `train.py`:
- `num_epochs`: Number of training epochs (default: 200)
- `batch_size`: Batch size (default: 64)
- `lr`: Learning rate (default: 0.0002)
- `latent_dim`: Size of latent vector (default: 100)
- `img_size`: Image size (default: 64x64)

### Generating Images

After training, generate images using the trained model:

#### Generate a grid of images:
```bash
python generate.py --checkpoint checkpoints/checkpoint_epoch_final.pth --num_images 64 --output my_pokemon.png
```

#### Generate individual images:
```bash
python generate.py --checkpoint checkpoints/checkpoint_epoch_final.pth --mode single --num_images 10 --output_dir my_generated_pokemon
```

#### Generate interpolation between latent vectors:
```bash
python generate.py --checkpoint checkpoints/checkpoint_epoch_final.pth --mode interpolate --num_images 10 --output interpolation.png
```

## Model Architecture

### Generator
- Input: Random noise vector (latent_dim, default: 100)
- Architecture: 5 transposed convolutional layers with batch normalization and ReLU
- Output: RGB image (3, 64, 64) with values in range [-1, 1]

### Discriminator
- Input: RGB image (3, 64, 64)
- Architecture: 5 convolutional layers with batch normalization and LeakyReLU
- Output: Single value representing probability that input is real

## Training Details

- **Loss Function**: Binary Cross-Entropy (BCE)
- **Optimizer**: Adam (lr=0.0002, beta1=0.5)
- **Batch Size**: 64
- **Image Size**: 64x64 pixels
- **Normalization**: Images normalized to [-1, 1]

## Tips for Better Results

1. **Train for more epochs**: GANs typically need 200+ epochs for good results
2. **Monitor losses**: Both Generator and Discriminator losses should stabilize over time
3. **Adjust learning rate**: If training is unstable, try reducing the learning rate
4. **Check generated samples**: Review samples saved during training to monitor progress
5. **Use GPU**: Training on GPU significantly speeds up the process

## Dataset

The project uses Pokemon sprite images from the `images/` directory. The dataset contains over 800 Pokemon images that are automatically loaded and preprocessed by the `PokemonDataset` class.

## Output

During training:
- Generated image samples are saved every 10 epochs to `generated_images/`
- Model checkpoints are saved to `checkpoints/`
- Loss plot is saved after training completion

## Requirements

- Python 3.7+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

## License

This project is for educational purposes.

## Acknowledgments

- DCGAN paper: [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)
- Pokemon sprites dataset
