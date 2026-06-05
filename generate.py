import torch
import matplotlib.pyplot as plt
import os
from models.vae import VAE


def generate_random_digits():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    latent_dims = [2, 5, 10, 20]
    num_samples = 100  # 10x10 grid = 100 generated digits

    os.makedirs("results", exist_ok=True)

    for latent_dim in latent_dims:
        print(f"Generating samples for {latent_dim}-D latent space...")

        checkpoint_path = f"checkpoints/aevb_mnist_{latent_dim}d.pth"

        # Check if the model weights exist before trying to load
        if not os.path.exists(checkpoint_path):
            print(f"  -> Skipping: Checkpoint {checkpoint_path} not found.")
            continue

        # Initialize model from checkpoints
        model = VAE(input_dim=784, latent_dim=latent_dim).to(device)
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True, map_location=device))
        model.eval()  # Set to evaluation mode

        # Random sampling from prior
        with torch.no_grad():
            # Shape: [100 images, latent_dim]
            sample_z = torch.randn(num_samples, latent_dim).to(device)

            # Pass through decoder
            generated_images = model.decoder(sample_z)

            # Reshape from vectors of 784 to images of 28x28
            generated_images = generated_images.view(num_samples, 28, 28).cpu().numpy()

        # Visualization (10x10 Grid)
        fig, axes = plt.subplots(10, 10, figsize=(8, 8))
        for i, ax in enumerate(axes.flatten()):
            # Using 'gray_r' to render black digits on a white background
            ax.imshow(generated_images[i], cmap='gray_r')
            ax.axis('off')

        plt.subplots_adjust(wspace=0.05, hspace=0.05)

        save_path = f"results/random_digits_{latent_dim}d.png"

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)


if __name__ == "__main__":
    generate_random_digits()
