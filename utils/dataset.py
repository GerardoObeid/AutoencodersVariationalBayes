import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset


class MNIST(Dataset):
    """
    A custom wrapper for the MNIST dataset to match the VAE paper's requirements.
    It automatically downloads the data, converts it to tensors,
    and flattens the 28x28 images into 784-dimensional vectors.
    """
    def __init__(self, root_dir='./data', train=True, download=True):
        # Define transformations:
        # 1. ToTensor() converts PIL image to PyTorch tensor and normalizes pixels to [0.0, 1.0]
        # 2. Lambda binarizes: pixels > 0.5 become 1, else 0 (Kingma & Welling paper uses binary MNIST)
        # 3. Lambda flattens the 2D spatial dimensions (28x28) into a 1D vector (784)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x > 0.5).float()),
            transforms.Lambda(lambda x: torch.flatten(x))
        ])

        # Load the standard MNIST dataset from torchvision
        self.mnist_data = datasets.MNIST(
            root=root_dir,
            train=train,
            transform=self.transform,
            download=download
        )

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.mnist_data)

    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset.
        Returns a tuple of (image_vector, label).
        """
        return self.mnist_data[idx]
