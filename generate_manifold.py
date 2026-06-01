import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from models.vae import VAE

def plot_manifold():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model configured for 2D latent space
    latent_dim = 2
    model = VAE(input_dim=784, latent_dim=latent_dim).to(device)
    model.load_state_dict(torch.load("checkpoints/vae_mnist_2d.pth", weights_only=True))
    model.eval()

    # Create the canvas for the 20x20 grid of images
    n = 20
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))

    # Generate X and Y axes using the inverse CDF (norm.ppf)
    # This distributes points based on the probability of a standard normal N(0,1)
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

    with torch.no_grad():
        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                # Take a pair of coordinates (x, y) from the 2D latent space
                z_sample = torch.tensor([[xi, yi]], dtype=torch.float32).to(device)

                # Pass through decoder only
                x_decoded = model.decoder(z_sample)

                # Reshape to 28x28
                digit = x_decoded.view(digit_size, digit_size).cpu().numpy()

                # Place the digit at the corresponding position on the canvas
                figure[i * digit_size: (i + 1) * digit_size,
                       j * digit_size: (j + 1) * digit_size] = digit

    # Plot the result
    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='gray_r')
    plt.axis('off')
    plt.title("2D Manifold of Latent Space (VAE MNIST)", fontsize=16)
    plt.tight_layout()
    plt.savefig("results/manifold_vae_mnist.png", dpi=300)

if __name__ == "__main__":
    plot_manifold()
