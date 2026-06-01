import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
import os


class FreyFace(Dataset):
    """Frey Face dataset - 1965 28x20 grayscale face images."""

    def __init__(self, root_dir='./data', train=True, download=True):
        self.root_dir = root_dir
        self.train = train
        self.mat_file = os.path.join(root_dir, 'frey_face','frey_rawface.mat')
        self.data_file = os.path.join(root_dir, 'frey_face.npy')

        if not os.path.exists(self.data_file):
            if not os.path.exists(self.mat_file):
                if download:
                    self._download_instructions()
                else:
                    raise FileNotFoundError(
                        f"Frey Face dataset not found. "
                    )
            self._load_from_mat()

        # Load dataset
        data = np.load(self.data_file)  # Shape: (1965, 560)

        # Normalize to [0, 1]
        data = data.astype(np.float32)
        if data.max() > 1:
            data = data / 255.0

        # --- THE FIX: Deterministic shuffle before splitting ---
        # Using a fixed seed ensures the train/test split doesn't overlap
        indices = np.random.RandomState(42).permutation(len(data))
        shuffled_data = data[indices]

        # 80/20 split
        n_train = int(0.8 * len(shuffled_data))
        if train:
            self.data = shuffled_data[:n_train]
        else:
            self.data = shuffled_data[n_train:]

    def _load_from_mat(self):
        """Load .mat file and save as .npy for faster loading."""
        print(f"Loading Frey Face from {self.mat_file}...")
        try:
            import time
            start = time.time()
            mat = sio.loadmat(self.mat_file)
            data = mat['ff'].T  # Transpose to get (1965, 560)
            data = data.astype(np.float32)

            # Save with compression for faster I/O
            np.save(self.data_file, data)
            elapsed = time.time() - start
            print(f"✓ Loaded and cached in {elapsed:.2f}s to {self.data_file}")
        except Exception as e:
            raise RuntimeError(f"Failed to load .mat file: {e}")

    def _download_instructions(self):
        """Print download instructions."""
        print("\n" + "="*70)
        print("FREY FACE DATASET NOT FOUND")
        print("="*70)
        print("Please download from Kaggle:")
        print("  https://www.kaggle.com/datasets/vineetvermaai/frey-face-dataset")
        print(f"Extract frey_rawface.mat to: {os.path.join(self.root_dir, 'frey_face/')}")
        print("="*70 + "\n")
        raise FileNotFoundError(
            f"Frey Face dataset (.mat file) not found at {self.mat_file}"
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]), 0


def get_frey_face_dataloaders(batch_size=128, data_dir='./data'):
    """
    Returns training and test DataLoaders for Frey Face.

    Args:
        batch_size (int): Minibatch size.
        data_dir (str): Directory where the dataset is stored/loaded from.

    Returns:
        tuple: (train_loader, test_loader)
    """
    train_dataset = FreyFace(root_dir=data_dir, train=True)
    test_dataset = FreyFace(root_dir=data_dir, train=False)

    # Frey Face is small (~1500 samples), so use fewer workers
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    return train_loader, test_loader
