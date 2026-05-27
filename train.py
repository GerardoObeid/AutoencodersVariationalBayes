import torch
import torch.optim as optim
from models.vae import VAE
from models.loss import vae_loss
from utils.get_dataset import get_mnist_dataloaders
import os
from tqdm import tqdm


def train(**kwargs):
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs = 50
    # To run pass as keyword as: python train.py --latent_dim 20
    latent_dim = kwargs.get("latent_dim")

    print(f"Training on: {device} for {epochs} epochs with latent dimension: {latent_dim}")
    # Initialize components
    train_loader, _ = get_mnist_dataloaders(batch_size=128)
    model = VAE(input_dim=784, latent_dim=latent_dim).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()

    # 3. Training loop and implemented tqdm to visualize the training progress
    step = 1
    train_loss = 0
    pbar = tqdm(range(1, epochs + 1))
    for epoch in pbar:
        pbar.set_description(f"Epoch {epoch}/{epochs} Step {step} loss: {train_loss/len(train_loader.dataset):.4f}")
        train_loss = 0
        # To get the trainig loss we need to divide the total loss by the number of samples in the dataset
        # since we are summing the loss over all batches and not averaging it.

        # We iterate over the minibatches where the paper mentions that needs to be M >= 100
        for batch_idx, (data, _) in enumerate(train_loader):
            step += 1
            data = data.to(device)

            optimizer.zero_grad()

            # Forward pass: L=1
            recon_batch, mu, log_var = model(data)

            # Calculamos la pérdida
            loss, bce, kld = vae_loss(recon_batch, data, mu, log_var)

            # Backward pass (Backpropagation with the reparameterization trick implemented at models/vae.py)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

    # 4. Save the trained model
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), f"checkpoints/vae_mnist_{latent_dim}d.pth")
    print(f"Trained model finished and saved\nTraining ELBO: {train_loss/len(train_loader.dataset):.4f}!")


if __name__ == "__main__":
    # To run pass as keyword as: python train.py --latent_dim 20
    import argparse
    parser = argparse.ArgumentParser(description="Train a VAE on MNIST with a specified latent dimension.")
    parser.add_argument('--latent_dim', type=int, default=2, help='Dimensionality of the latent space (default: 20)')
    args = parser.parse_args()
    train(**vars(args))
