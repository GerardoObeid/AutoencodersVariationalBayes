import torch.nn as nn
import torch

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=500, dataset='mnist'):
        super(VAE, self).__init__()
        self.dataset = dataset
        
        # ==========================================
        # 1. THE ENCODER (Shared by both)
        # ==========================================
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)

        # ==========================================
        # 2. THE DECODER (Dataset Specific)
        # ==========================================
        if self.dataset == 'mnist':
            # Bernoulli Decoder (Section C.1)
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, input_dim),
                nn.Sigmoid()
            )
            
        elif self.dataset == 'frey_face':
            # Gaussian Decoder (Section C.2)
            self.decoder_hidden = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.Tanh()
            )
            self.dec_mu = nn.Sequential(
                nn.Linear(hidden_dim, input_dim),
                nn.Sigmoid()
            )    
            self.dec_log_var = nn.Linear(hidden_dim, input_dim)
            
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.normal_(m.bias, mean=0.0, std=0.01)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # ==========================================
        # 1. Encode
        # ==========================================
        h = self.encoder(x)
        mu_z = self.fc_mu(h)
        log_var_z = self.fc_log_var(h)


        # ==========================================
        # 2. Sample
        # ==========================================
        z = self.reparameterize(mu_z, log_var_z)

        # ==========================================
        # 3. Decode based on dataset
        # ==========================================
        if self.dataset == 'mnist':
            # Returns a single tensor (probabilities)
            recon_x = self.decoder(z)
            return recon_x, mu_z, log_var_z
            
        elif self.dataset == 'frey_face':
            # Returns two tensors (mean and log variance)
            h_dec = self.decoder_hidden(z)
            recon_mu = self.dec_mu(h_dec)
            recon_log_var = self.dec_log_var(h_dec)
            
            # Pack the two outputs into a tuple to keep the return signature clean
            return (recon_mu, recon_log_var), mu_z, log_var_z