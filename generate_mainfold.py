import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from models.vae import VAE

def plot_manifold():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # IMPORTANTE: Cargamos el modelo configurado para 2 dimensiones
    latent_dim = 2
    model = VAE(input_dim=784, latent_dim=latent_dim).to(device)
    model.load_state_dict(torch.load("checkpoints/vae_mnist_2d.pth", weights_only=True))
    model.eval()

    # Configuración del lienzo
    n = 20  # Crearemos una cuadrícula de 20x20 imágenes
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))

    # Generamos los ejes X e Y usando la inversa de la CDF (norm.ppf)
    # Esto distribuye los puntos basándose en la probabilidad de una Normal N(0,1)
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

    with torch.no_grad():
        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                # Tomamos un par de coordenadas (x, y) del espacio latente 2D
                z_sample = torch.tensor([[xi, yi]], dtype=torch.float32).to(device)
                
                # Lo pasamos SOLO por el Decoder
                x_decoded = model.decoder(z_sample)
                
                # Redimensionamos a 28x28
                digit = x_decoded.view(digit_size, digit_size).cpu().numpy()
                
                # Pegamos el dígito en la posición correspondiente del lienzo gigante
                figure[i * digit_size: (i + 1) * digit_size,
                       j * digit_size: (j + 1) * digit_size] = digit

    # Dibujamos el resultado
    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='gray')
    plt.axis('off')
    plt.title("2D Manifold of Latent Space (VAE MNIST)", fontsize=16)
    plt.tight_layout()
    #plt.show()
    # Save figure
    plt.savefig("results/manifold_vae_mnist.png", dpi=300)

if __name__ == "__main__":
    plot_manifold()