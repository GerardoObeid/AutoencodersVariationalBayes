import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
# Ensure your VAE class handles the input_dim parameter correctly
from models.vae import VAE 

def plot_frey_manifold():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model configured for 2D latent space and Frey Face input
    latent_dim = 2
    model = VAE(input_dim=560, latent_dim=latent_dim, hidden_dim=200, dataset='frey_face').to(device)
    
    # IMPORTANT: Update this to point to your actual Frey Face .pth file
    model.load_state_dict(torch.load("checkpoints/aevb_frey_face_2d.pth", weights_only=True))
    model.eval()

    # Create the canvas for the 20x20 grid of images
    n = 20
    height = 28
    width = 20
    
    # The canvas now needs distinct height and width multipliers
    figure = np.zeros((height * n, width * n))

    # Generate X and Y axes using the inverse CDF (norm.ppf)
    grid_x = norm.ppf(np.linspace(0.01, 0.99, n))
    grid_y = norm.ppf(np.linspace(0.01, 0.99, n))

    with torch.no_grad():
        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                # Take a pair of coordinates (x, y) from the 2D latent space
                z_sample = torch.tensor([[xi, yi]], dtype=torch.float32).to(device)

                # Pass through decoder only
                h_dec = model.decoder_hidden(z_sample)
                x_decoded = model.dec_mu(h_dec)

                # Reshape to 28x20
                face = x_decoded.view(height, width).cpu().numpy()
                
                # Note: Depending on how the original .mat file was flattened, 
                # if the faces look scrambled or rotated 90 degrees, replace the line above with:
                # face = x_decoded.cpu().numpy().reshape(height, width, order='F')

                # Place the face at the corresponding position on the canvas
                figure[i * height: (i + 1) * height,
                       j * width: (j + 1) * width] = face

    # Plot the result
    # Adjusted figsize to reflect the 28:20 (or 1.4:1) height-to-width ratio
    # Find the actual min and max values the Sigmoid dared to output
    canvas_min = figure.min()
    canvas_max = figure.max()
    
    # Mathematically stretch those values to span from exactly 0.0 to 1.0
    figure = (figure - canvas_min) / (canvas_max - canvas_min + 1e-8)
    # ------------------------------------------

    # Plot the result
    plt.figure(figsize=(10, 14)) 
    plt.imshow(figure, cmap='gray', vmin=0, vmax=1) # Lock the display to our stretched bounds
    plt.axis('off')
    plt.title("2D Manifold of Latent Space (VAE Frey Face)", fontsize=16)
    plt.tight_layout()
    plt.savefig("results/manifold_vae_freyface.png", dpi=300)

if __name__ == "__main__":
    plot_frey_manifold()