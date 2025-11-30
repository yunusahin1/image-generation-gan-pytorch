import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image
from PIL import Image
import os


class PokemonDataset(Dataset):
    """
    Custom Dataset for loading Pokemon images
    """
    def __init__(self, root_dir, img_size=64, transform=None):
        """
        Args:
            root_dir (str): Directory with all the Pokemon images
            img_size (int): Size to resize images to (default: 64x64)
            transform (callable, optional): Optional transform to be applied on images
        """
        self.root_dir = root_dir
        self.img_size = img_size
        
        # Get all image files
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.png')]
        
        # Define transforms if not provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
            ])
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        
        # Load image using PIL (more reliable than torchvision.io)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image


def get_dataloader(root_dir='images', batch_size=32, img_size=64, num_workers=2, shuffle=True):
    """
    Create a DataLoader for the Pokemon dataset
    
    Args:
        root_dir (str): Directory containing Pokemon images
        batch_size (int): Number of images per batch
        img_size (int): Size to resize images to
        num_workers (int): Number of worker processes for data loading
        shuffle (bool): Whether to shuffle the data
    
    Returns:
        DataLoader: PyTorch DataLoader object
    """
    dataset = PokemonDataset(root_dir=root_dir, img_size=img_size)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader