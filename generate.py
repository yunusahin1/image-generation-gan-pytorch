import torch
from torchvision.utils import save_image
import os
import argparse

from generator import Generator


def load_generator(checkpoint_path, latent_dim=100, img_channels=3, feature_maps=64, device='cpu'):
    """
    Load a trained generator from checkpoint
    
    Args:
        checkpoint_path (str): Path to the checkpoint file
        latent_dim (int): Latent dimension size
        img_channels (int): Number of image channels
        feature_maps (int): Number of feature maps
        device (str): Device to load model on
    
    Returns:
        Generator model
    """
    generator = Generator(latent_dim=latent_dim, img_channels=img_channels, feature_maps=feature_maps)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.to(device)
    generator.eval()
    
    print(f"Generator loaded from {checkpoint_path}")
    if isinstance(checkpoint['epoch'], int):
        print(f"Trained for {checkpoint['epoch']} epochs")
    
    return generator


def generate_images(generator, num_images=64, latent_dim=100, output_path='generated.png', device='cpu'):
    """
    Generate images using a trained generator
    
    Args:
        generator: Trained Generator model
        num_images (int): Number of images to generate
        latent_dim (int): Latent dimension size
        output_path (str): Path to save generated images
        device (str): Device to use for generation
    """
    with torch.no_grad():
        # Generate random noise
        noise = torch.randn(num_images, latent_dim, 1, 1, device=device)
        
        # Generate fake images
        fake_images = generator(noise)
        
        # Denormalize from [-1, 1] to [0, 1]
        fake_images = (fake_images + 1) / 2
        
        # Save images in a grid
        save_image(fake_images, output_path, nrow=8, normalize=False)
        
        print(f"Generated {num_images} images saved to {output_path}")


def generate_single_images(generator, num_images=10, latent_dim=100, output_dir='generated_single', device='cpu'):
    """
    Generate individual image files
    
    Args:
        generator: Trained Generator model
        num_images (int): Number of images to generate
        latent_dim (int): Latent dimension size
        output_dir (str): Directory to save individual images
        device (str): Device to use for generation
    """
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        for i in range(num_images):
            # Generate random noise for one image
            noise = torch.randn(1, latent_dim, 1, 1, device=device)
            
            # Generate image
            fake_image = generator(noise)
            
            # Denormalize from [-1, 1] to [0, 1]
            fake_image = (fake_image + 1) / 2
            
            # Save individual image
            output_path = os.path.join(output_dir, f'generated_{i+1}.png')
            save_image(fake_image, output_path, normalize=False)
        
        print(f"Generated {num_images} individual images in {output_dir}/")


def interpolate_latent_space(generator, num_steps=10, latent_dim=100, output_path='interpolation.png', device='cpu'):
    """
    Generate images by interpolating between two random points in latent space
    
    Args:
        generator: Trained Generator model
        num_steps (int): Number of interpolation steps
        latent_dim (int): Latent dimension size
        output_path (str): Path to save interpolated images
        device (str): Device to use for generation
    """
    with torch.no_grad():
        # Generate two random latent vectors
        z1 = torch.randn(1, latent_dim, 1, 1, device=device)
        z2 = torch.randn(1, latent_dim, 1, 1, device=device)
        
        # Interpolate between z1 and z2
        alphas = torch.linspace(0, 1, num_steps, device=device)
        interpolated_images = []
        
        for alpha in alphas:
            z = (1 - alpha) * z1 + alpha * z2
            fake_image = generator(z)
            interpolated_images.append(fake_image)
        
        # Concatenate all images
        interpolated_images = torch.cat(interpolated_images, dim=0)
        
        # Denormalize from [-1, 1] to [0, 1]
        interpolated_images = (interpolated_images + 1) / 2
        
        # Save images
        save_image(interpolated_images, output_path, nrow=num_steps, normalize=False)
        
        print(f"Interpolation with {num_steps} steps saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate images using trained GAN')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/checkpoint_epoch_final.pth',
                        help='Path to checkpoint file')
    parser.add_argument('--num_images', type=int, default=64,
                        help='Number of images to generate')
    parser.add_argument('--output', type=str, default='generated_output.png',
                        help='Output path for generated images')
    parser.add_argument('--mode', type=str, default='grid', choices=['grid', 'single', 'interpolate'],
                        help='Generation mode: grid, single, or interpolate')
    parser.add_argument('--output_dir', type=str, default='generated_single',
                        help='Output directory for single images')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='Latent dimension size')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load generator
    generator = load_generator(
        args.checkpoint,
        latent_dim=args.latent_dim,
        device=device
    )
    
    # Generate images based on mode
    if args.mode == 'grid':
        generate_images(
            generator,
            num_images=args.num_images,
            latent_dim=args.latent_dim,
            output_path=args.output,
            device=device
        )
    elif args.mode == 'single':
        generate_single_images(
            generator,
            num_images=args.num_images,
            latent_dim=args.latent_dim,
            output_dir=args.output_dir,
            device=device
        )
    elif args.mode == 'interpolate':
        interpolate_latent_space(
            generator,
            num_steps=args.num_images,
            latent_dim=args.latent_dim,
            output_path=args.output,
            device=device
        )
