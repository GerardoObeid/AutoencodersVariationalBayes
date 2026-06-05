from torch.utils.data import DataLoader
from .dataset import MNIST


def get_mnist_dataloaders(batch_size=100, data_dir='./data'):
    """
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
