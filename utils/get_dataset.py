from torch.utils.data import DataLoader
from .dataset import MNIST

def get_mnist_dataloaders(batch_size=100, data_dir='./data'):
    """
    Returns the training and testing DataLoaders for the VAE.

    PAPER REFERENCE: 
    Auto-Encoding Variational Bayes (Kingma & Welling, 2013) 
    Section 2.3, Section 4 (Experiments), and Algorithm 1 (Minibatch AEVB).

    THE BATCH SIZE (M) AND NUMBER OF SAMPLES (L) RELATIONSHIP:
    In variational inference, estimating the Evidence Lower Bound (ELBO) technically 
    requires taking multiple samples (L) from the approximate posterior for each 
    individual datapoint to get an accurate gradient.
    
    However, the authors' core practical finding is that if the minibatch size (M) 
    is large enough, the inherent noise across the batch acts as a sufficient 
    regularizer. Therefore:
    
    - M = 100: The paper explicitly uses a standard minibatch size of 100 datapoints 
      for the MNIST experiments.
    - L = 1: Because M (100) is sufficiently large, we can safely set the number of 
      latent samples per datapoint to L=1. This means we only perform the 
      reparameterization trick ONCE per image during the forward pass. 
      This eliminates the need for expensive repeated sampling, making the 
      algorithm highly computationally efficient and scalable.

    Args:
        batch_size (int): Minibatch size (M). Defaults to 100 as per the paper.
        data_dir (str): Directory where the dataset will be stored/loaded from.

    Returns:
        tuple: (train_loader, test_loader)
    """
    
    # Initialize the custom dataset classes (downloads if not present)
    train_dataset = MNIST(root_dir=data_dir, train=True)
    test_dataset = MNIST(root_dir=data_dir, train=False)

    # Create the DataLoaders
    # We shuffle the training data to ensure stochasticity in gradient descent updates
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    # Shuffling is not strictly necessary for the test set
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader