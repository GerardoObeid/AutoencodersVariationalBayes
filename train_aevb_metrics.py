import torch
import torch.optim as optim
from models.vae import VAE
from models.loss import vae_loss
from utils.get_dataset import get_mnist_dataloaders
from utils.frey_face import get_frey_face_dataloaders
import os
import json
from tqdm import tqdm


# Change the function signature to include weight_decay
def train_aevb(dataset='mnist', latent_dim=20, epochs=50, batch_size=128, lr=1e-3, save_metrics=True):   
    """
    Train VAE with AEVB algorithm and record metrics during training.

    Args:
        dataset: 'mnist' or 'frey_face'
        latent_dim: Latent dimension
        epochs: Number of training epochs
        batch_size: Minibatch size
        lr: Learning rate
        save_metrics: Whether to save metrics during training

    Returns:
        dict: Training metrics (samples_evaluated, train_loss, test_loss)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset and define hidden dimensions based on Appendix C
    if dataset == 'mnist':
        train_loader, test_loader = get_mnist_dataloaders(batch_size=batch_size)
        input_dim = 784
        hidden_dim = 500
    elif dataset == 'frey_face':
        train_loader, test_loader = get_frey_face_dataloaders(batch_size=batch_size)
        input_dim = 560
        hidden_dim = 200
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    print(f"Training AEVB on {dataset} (Nz={latent_dim}) for {epochs} epochs on {device}")

    # Initialize model WITH the dataset and hidden_dim parameters
    model = VAE(
        input_dim=input_dim, 
        latent_dim=latent_dim, 
        hidden_dim=hidden_dim, 
        dataset=dataset
    ).to(device)

    # Adagrad and add weight decay (L2 penalty) representing the N(0, I) prior
    optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=1e-4)

    # Metrics tracking
    metrics = {
        'samples_evaluated': [],
        'train_loss': [],
        'test_loss': [],
    }

    model.train()
    total_samples_seen = 0
    pbar = tqdm(range(1, epochs + 1))

    for epoch in pbar:
        epoch_loss = 0

        for data, _ in train_loader:
            data = data.to(device)
            current_batch_size = data.size(0)

            optimizer.zero_grad()
            
            # recon_batch will be a single tensor (MNIST) or a tuple (Frey Face)
            recon_batch, mu, log_var = model(data)
            
            # The loss function will correctly unpack the tuple for Frey Face
            loss, _, _ = vae_loss(recon_batch, data, mu, log_var, dataset=dataset)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            total_samples_seen += current_batch_size

        # Evaluate on test set
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(device)
                recon_batch, mu, log_var = model(data)
                loss, _, _ = vae_loss(recon_batch, data, mu, log_var, dataset=dataset)
                test_loss += loss.item()

        model.train()

        # Normalize losses
        train_loss_avg = epoch_loss / len(train_loader.dataset)
        test_loss_avg = test_loss / len(test_loader.dataset)

        metrics['samples_evaluated'].append(total_samples_seen)
        metrics['train_loss'].append(train_loss_avg)
        metrics['test_loss'].append(test_loss_avg)

        pbar.set_description(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_loss_avg:.4f} | Test Loss: {test_loss_avg:.4f} | Samples: {total_samples_seen}"
        )

    # Save model
    os.makedirs("checkpoints", exist_ok=True)
    model_path = f"checkpoints/aevb_{dataset}_{latent_dim}d.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Save metrics
    if save_metrics:
        metrics_path = f"results/metrics/aevb_{dataset}_{latent_dim}d_metrics.json"
        os.makedirs("results/metrics", exist_ok=True)
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f)
        print(f"Metrics saved to {metrics_path}")

    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train AEVB with metrics tracking")
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'frey_face'])
    parser.add_argument('--latent_dim', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()

    train_aevb(
        dataset=args.dataset,
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )