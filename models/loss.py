import math

import torch
import torch.nn.functional as F

def vae_loss(recon_output, x, mu, log_var, dataset='mnist'):
    """
    Compute the VAE loss (negative Evidence Lower Bound - ELBO).

    The loss consists of two parts:
        1. Reconstruction Error: Measures how well the decoder reconstructs
           the input data (Bernoulli for MNIST, Gaussian for Frey Face).
        2. Kullback-Leibler Divergence (KLD): Measures how closely the learned
           latent distribution matches the prior.

    Args:
        recon_x: The reconstructed output from the decoder.
        x: The original input data (flattened).
        mu: The mean of the latent distribution (encoder output).
        log_var: The log variance of the latent distribution (encoder output).
        dataset: String indicating which observation model to use.
    """

    # Kullback-Leibler Divergence (KLD)
    # Closed-form solution from Kingma & Welling (2013).
    # This remains exactly the same for all AEVB models.
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    if dataset == 'mnist':
        # recon_output is just the single tensor here
        recon_loss = F.binary_cross_entropy(recon_output, x, reduction='sum')

    elif dataset == 'frey_face':
        # Unpack the tuple
        recon_mu, recon_log_var = recon_output

        sq_err = (recon_mu - x).pow(2)
        recon_loss = 0.5 * torch.sum(math.log(2 * math.pi) + recon_log_var + sq_err / recon_log_var.exp())
        
    total_loss = recon_loss + KLD
    return total_loss, recon_loss, KLD