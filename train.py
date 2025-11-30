import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

from generator import Generator, weights_init
from discriminator import Discriminator
from dataloader import get_dataloader


class GANTrainer:
    """
    GAN Training Class
    """
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create directories
        os.makedirs(config['output_dir'], exist_ok=True)
        os.makedirs(config['checkpoint_dir'], exist_ok=True)
        
        # Initialize models
        self.generator = Generator(
            latent_dim=config['latent_dim'],
            img_channels=config['img_channels'],
            feature_maps=config['feature_maps']
        ).to(self.device)
        
        self.discriminator = Discriminator(
            img_channels=config['img_channels'],
            feature_maps=config['feature_maps']
        ).to(self.device)
        
        # Initialize weights
        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        # Optimizers
        self.optimizer_G = optim.Adam(
            self.generator.parameters(),
            lr=config['lr'],
            betas=(config['beta1'], 0.999)
        )
        
        self.optimizer_D = optim.Adam(
            self.discriminator.parameters(),
            lr=config['lr'],
            betas=(config['beta1'], 0.999)
        )
        
        # For tracking
        self.G_losses = []
        self.D_losses = []
        self.fixed_noise = torch.randn(64, config['latent_dim'], 1, 1, device=self.device)
        
    def train_epoch(self, dataloader, epoch):
        """Train for one epoch"""
        self.generator.train()
        self.discriminator.train()
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for i, real_images in enumerate(progress_bar):
            batch_size = real_images.size(0)
            real_images = real_images.to(self.device)
            
            # Create labels
            real_labels = torch.ones(batch_size, 1, 1, 1, device=self.device)
            fake_labels = torch.zeros(batch_size, 1, 1, 1, device=self.device)
            
            # =================
            # Train Discriminator
            # =================
            self.optimizer_D.zero_grad()
            
            # Real images
            output_real = self.discriminator(real_images)
            loss_D_real = self.criterion(output_real, real_labels)
            
            # Fake images
            noise = torch.randn(batch_size, self.config['latent_dim'], 1, 1, device=self.device)
            fake_images = self.generator(noise)
            output_fake = self.discriminator(fake_images.detach())
            loss_D_fake = self.criterion(output_fake, fake_labels)
            
            # Total discriminator loss
            loss_D = loss_D_real + loss_D_fake
            loss_D.backward()
            self.optimizer_D.step()
            
            # =================
            # Train Generator
            # =================
            self.optimizer_G.zero_grad()
            
            # Generate fake images and try to fool discriminator
            output_fake = self.discriminator(fake_images)
            loss_G = self.criterion(output_fake, real_labels)
            
            loss_G.backward()
            self.optimizer_G.step()
            
            # Update progress bar
            progress_bar.set_postfix({
                'D_loss': f'{loss_D.item():.4f}',
                'G_loss': f'{loss_G.item():.4f}'
            })
            
            # Save losses
            self.G_losses.append(loss_G.item())
            self.D_losses.append(loss_D.item())
        
        return loss_D.item(), loss_G.item()
    
    def train(self, num_epochs):
        """Train the GAN"""
        dataloader = get_dataloader(
            root_dir=self.config['data_dir'],
            batch_size=self.config['batch_size'],
            img_size=self.config['img_size'],
            num_workers=self.config['num_workers']
        )
        
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Total images: {len(dataloader.dataset)}")
        print(f"Batch size: {self.config['batch_size']}")
        print(f"Batches per epoch: {len(dataloader)}\n")
        
        for epoch in range(1, num_epochs + 1):
            loss_D, loss_G = self.train_epoch(dataloader, epoch)
            
            print(f"Epoch [{epoch}/{num_epochs}] - D_loss: {loss_D:.4f}, G_loss: {loss_G:.4f}")
            
            # Save generated images
            if epoch % self.config['save_interval'] == 0:
                self.save_samples(epoch)
                self.save_checkpoint(epoch)
        
        # Save final model
        self.save_checkpoint('final')
        self.plot_losses()
        
        print("\nTraining completed!")
    
    def save_samples(self, epoch):
        """Generate and save sample images"""
        self.generator.eval()
        with torch.no_grad():
            fake_images = self.generator(self.fixed_noise)
            # Denormalize from [-1, 1] to [0, 1]
            fake_images = (fake_images + 1) / 2
            
            save_path = os.path.join(self.config['output_dir'], f'epoch_{epoch}.png')
            save_image(fake_images, save_path, nrow=8, normalize=False)
        self.generator.train()
    
    def save_checkpoint(self, epoch):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'G_losses': self.G_losses,
            'D_losses': self.D_losses,
        }
        
        save_path = os.path.join(self.config['checkpoint_dir'], f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved: {save_path}")
    
    def plot_losses(self):
        """Plot training losses"""
        plt.figure(figsize=(10, 5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(self.G_losses, label="Generator", alpha=0.7)
        plt.plot(self.D_losses, label="Discriminator", alpha=0.7)
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(self.config['output_dir'], 'loss_plot.png'))
        plt.close()


if __name__ == "__main__":
    # Training configuration
    config = {
        'data_dir': 'images',
        'output_dir': 'generated_images',
        'checkpoint_dir': 'checkpoints',
        'latent_dim': 100,
        'img_channels': 3,
        'img_size': 64,
        'feature_maps': 64,
        'batch_size': 64,
        'num_epochs': 10000,
        'lr': 0.0002,
        'beta1': 0.5,
        'num_workers': 2,
        'save_interval': 10
    }
    
    # Create trainer and start training
    trainer = GANTrainer(config)
    trainer.train(config['num_epochs'])
