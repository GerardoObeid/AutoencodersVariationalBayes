import torch
import torch.nn.functional as F

def vae_loss(recon_x, x, mu, log_var):
    """
    Calcula la pérdida del VAE (Negative ELBO).
    """
    # 1. Error de Reconstrucción (BCE)
    # Usamos reduction='sum' porque queremos la suma sobre todos los píxeles (784) 
    # y sobre todo el batch, como dicta la teoría probabilística.
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # 2. Divergencia Kullback-Leibler (KLD)
    # Fórmula cerrada analítica del paper: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    # La pérdida total que queremos minimizar es la suma de ambas
    total_loss = BCE + KLD
    
    return total_loss, BCE, KLD