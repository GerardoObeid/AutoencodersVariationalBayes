"""
Entrenamiento Wake-Sleep (Hinton et al., 1995) sobre MNIST.

Misma arquitectura VAE que train.py, pero sin backprop conjunta del ELBO:
  - Fase wake (despierto): datos reales x -> q_phi(z|x) -> p_theta(x|z).
    Solo se actualiza el modelo generativo p_theta (decoder).
  - Fase sleep (sueño): z ~ N(0,I) -> p_theta(x|z) -> x_onirico;
    luego q_phi aprende a inferir z a partir de x_onirico.
    Solo se actualiza el modelo de reconocimiento q_phi (encoder + cabezas).

Sirve como baseline frente a AEVB (Kingma & Welling), que optimiza encoder y
decoder a la vez con el truco de reparametrización.
"""

import argparse
import math
import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from models.loss import vae_loss
from models.vae import VAE
from utils.get_dataset import get_mnist_dataloaders


def gaussian_nll(z, mu, log_var):
    """
    Log-verosimilitud negativa de z bajo q_phi(z|x) = N(mu, diag(exp(log_var))).

    En sleep, z viene del prior y mu/log_var del encoder sobre datos soñados;
    minimizar esto acerca q_phi al prior cuando x es generado por el decoder.
    """
    return 0.5 * torch.sum(
        math.log(2 * math.pi) + log_var + (z - mu).pow(2) / log_var.exp()
    )


def evaluate_elbo(model, data_loader, device):
    """
    NELBO medio en el conjunto (misma pérdida que train.py / vae_loss).

    No entrena; solo monitoriza si el modelo conjunto mejora aunque wake/sleep
    actualicen partes distintas por separado.
    """
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            recon_batch, mu, log_var = model(data)
            loss, _, _ = vae_loss(recon_batch, data, mu, log_var)
            total_loss += loss.item()

    model.train()
    return total_loss / len(data_loader.dataset)


def train_wake_sleep(**kwargs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs = kwargs.get("epochs")
    latent_dim = kwargs.get("latent_dim")
    batch_size = kwargs.get("batch_size")
    lr = kwargs.get("lr")
    sleep_batches_per_wake = kwargs.get("sleep_batches_per_wake")

    print(
        "Training Wake-Sleep on:"
        f" {device} for {epochs} epochs with latent dimension: {latent_dim}"
    )

    train_loader, test_loader = get_mnist_dataloaders(batch_size=batch_size)
    model = VAE(input_dim=784, latent_dim=latent_dim).to(device)

    # Optimizadores separados: cada fase solo toca un subconjunto de parámetros.
    generative_optimizer = optim.Adam(model.decoder.parameters(), lr=lr)
    recognition_optimizer = optim.Adam(
        list(model.encoder.parameters())
        + list(model.fc_mu.parameters())
        + list(model.fc_log_var.parameters()),
        lr=lr,
    )

    model.train()
    pbar = tqdm(range(1, epochs + 1))

    for epoch in pbar:
        wake_loss_total = 0
        sleep_loss_total = 0

        for data, _ in train_loader:
            data = data.to(device)
            current_batch_size = data.size(0)

            # --- Fase WAKE: aprender p_theta(x|z) con datos reales ---
            # q_phi(z|x) solo hace inferencia; sus gradientes no se propagan.
            with torch.no_grad():
                h = model.encoder(data)
                mu = model.fc_mu(h)
                log_var = model.fc_log_var(h)
                z = model.reparameterize(mu, log_var)

            generative_optimizer.zero_grad()
            recon_batch = model.decoder(z)
            # BCE sumada por minibatch (como en el paper, M=100).
            wake_loss = F.binary_cross_entropy(recon_batch, data, reduction="sum")
            wake_loss.backward()
            generative_optimizer.step()
            wake_loss_total += wake_loss.item()

            # --- Fase SLEEP: aprender q_phi(z|x) con fantasías del generativo ---
            # z ~ prior; el decoder produce probabilidades; muestreamos x binario.
            for _ in range(sleep_batches_per_wake):
                with torch.no_grad():
                    z_prior = torch.randn(current_batch_size, latent_dim).to(device)
                    dreamed_probs = model.decoder(z_prior)
                    dreamed_x = torch.bernoulli(dreamed_probs)

                recognition_optimizer.zero_grad()
                h_sleep = model.encoder(dreamed_x)
                mu_sleep = model.fc_mu(h_sleep)
                log_var_sleep = model.fc_log_var(h_sleep)
                # Ajustar q_phi para que asigne alta densidad al z que generó el sueño.
                sleep_loss = gaussian_nll(z_prior, mu_sleep, log_var_sleep)
                sleep_loss.backward()
                recognition_optimizer.step()
                sleep_loss_total += sleep_loss.item()

        # Métricas de referencia (ELBO completo) y pérdidas por fase.
        train_neg_elbo = evaluate_elbo(model, train_loader, device)
        test_neg_elbo = evaluate_elbo(model, test_loader, device)

        wake_avg = wake_loss_total / len(train_loader.dataset)
        sleep_avg = sleep_loss_total / (
            len(train_loader.dataset) * sleep_batches_per_wake
        )
        pbar.set_description(
            f"Epoch {epoch}/{epochs} "
            f"wake: {wake_avg:.4f} sleep: {sleep_avg:.4f} "
            f"train NELBO: {train_neg_elbo:.4f} test NELBO: {test_neg_elbo:.4f}"
        )

    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = f"checkpoints/wake_sleep_mnist_{latent_dim}d.pth"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Wake-Sleep model saved to {checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a Wake-Sleep model on MNIST for comparison with AEVB."
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=20,
        help="Dimensionality of the latent space.",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Minibatch size. The AEVB paper used M=100.",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument(
        "--sleep_batches_per_wake",
        type=int,
        default=1,
        help="Number of sleep updates after each wake update.",
    )
    args = parser.parse_args()
    train_wake_sleep(**vars(args))
