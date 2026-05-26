import torch
import torch.nn.functional as F


def vae_loss(recon_x, x, mu, log_var):
    """
    This script calculates the loss for the VAE (Negative ELBO).
    The loss consists of two parts:
        1. Reconstruction Error (BCE): Measures how well the decoder can reconstruct the input data.
        2. Kullback-Leibler Divergence (KLD): Measures how closely the learned latent distribution (parameterized by mu and log_var) 

    The paper "Auto-Encoding Variational Bayes" by Kingma and Welling (2013) shows that by increasing the latent space,
    the model doesn't overfit because of the regularization effect of the KL divergence. 
    The KL divergence encourages the latent space to be close to a standard normal distribution,
    which prevents the model from simply memorizing the training data and promotes generalization to unseen data.
    Args:
        recon_x: The reconstructed output from the decoder (after passing through sigmoid).
        x: The original input data (flattened).
        mu: The mean of the latent distribution (output from the encoder).
        log_var: The log variance of the latent distribution (output from the encoder).
    """

    # 1. Reconstruction Error (BCE)
    # We use reduction='sum' because we want the sum over all pixels (784) 
    # and over the entire batch, as dictated by the probabilistic theory.
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # 2. Kullback-Leibler Divergence (KLD)
    # Analytical closed-form formula from the paper: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    # The total loss we want to minimize is the sum of both
    total_loss = BCE + KLD
    
    return total_loss, BCE, KLD
