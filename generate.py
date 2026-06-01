import torch
import matplotlib.pyplot as plt
import os
from models.vae import VAE


def generate_random_digits():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Latent dimensions corresponding to the reference image
    latent_dims = [2, 5, 10, 20]
    num_samples = 100  # 10x10 grid 
    
    # Ensure the output directory exists
    os.makedirs("results", exist_ok=True)

    for latent_dim in latent_dims:
        print(f"Generating samples for {latent_dim}-D latent space...")
        
        checkpoint_path = f"checkpoints/aevb_mnist_{latent_dim}d.pth"
        
        # Check if the model weights exist before trying to load
        if not os.path.exists(checkpoint_path):
            print(f"  -> Skipping: Checkpoint {checkpoint_path} not found.")
            continue

        # Initialize and load the trained model
        model = VAE(input_dim=784, latent_dim=latent_dim).to(device)
        # Added map_location=device to prevent issues if moving weights between CPU/GPU
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

        # Reduce spacing between images to match the dense look of the reference
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        
        # Save each grid as a separate image
        save_path = f"results/random_digits_{latent_dim}d.png"
        
        # bbox_inches='tight' trims the excess white border around the plot
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)  # Close figure to free up memory before the next loop
        
        print(f"  -> Saved grid to {save_path}")

if __name__ == "__main__":
    generate_random_digits()