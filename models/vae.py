import torch.nn as nn
import torch

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        """Followind the VAE architecture from the original paper "Auto-Encoding Variational Bayes" by Kingma and Welling (2013)"
            We construct the encoder as a MLP with two hidden layers and the decoder as a MLP with two hidden layers as well. 
            The output of the encoder is split into two parts, one for the mean and one for the log variance of the latent space distribution. 
            The reparameterization trick is used to sample from the latent space during training."""
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.Tanh() # Output mean and log variance
        )

        self.fc_mu = nn.Linear(500, latent_dim)
        self.fc_log_var = nn.Linear(500, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 500),
            nn.Tanh(),
            nn.Linear(500, input_dim),
            nn.Sigmoid() # The input needs to be normalized between 0 and 1
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encoder's hidden layer
        h = self.encoder(x)
        
        # We obtain the parameters of the latent space distribution
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        
        # Reparameterization trick to sample from the latent space
        """
            Instead of sampling directly from the distribution defined by mu and log_var:
            z ~ N(mu, sigma^2)
            We sample from a standard normal distribution and then transform it using the parameters of our distribution:
            z = mu + sigma * eps
                Where eps ~ N(0, 1) and sigma = exp(0.5 * log_var)

        """
        z = self.reparameterize(mu, log_var)
        
        # 4. Reconstrucción
        decoded = self.decoder(z)
        
        return decoded, mu, log_var