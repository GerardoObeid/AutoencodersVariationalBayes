import torch
import torch.optim as optim
from models.vae import VAE
from models.loss import vae_loss
from utils.get_dataset import get_mnist_dataloaders
import os

def train():
    # 1. Configuración
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Entrenando en: {device}")
    
    epochs = 50
    latent_dim = 2 # Dimensión del espacio latente recomendada en el paper
    
    # 2. Inicializamos Componentes
    train_loader, _ = get_mnist_dataloaders(batch_size=100)
    model = VAE(input_dim=784, latent_dim=latent_dim).to(device)
    
    # Aunque el paper menciona Adagrad, Adam (2014) se convirtió en el estándar 
    # casi inmediatamente después para entrenar VAEs.
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    model.train() # Modo entrenamiento
    
    # 3. Bucle de Entrenamiento
    for epoch in range(1, epochs + 1):
        train_loss = 0
        
        # Iteramos sobre los mini-batches (M=100)
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            
            optimizer.zero_grad() # Limpiamos gradientes anteriores
            
            # Forward pass: L=1 (Un solo muestreo del espacio latente gracias al batch size)
            recon_batch, mu, log_var = model(data)
            
            # Calculamos la pérdida
            loss, bce, kld = vae_loss(recon_batch, data, mu, log_var)
            
            # Backward pass (Backpropagation a través del reparameterization trick!)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        # Imprimir progreso
        avg_loss = train_loss / len(train_loader.dataset)
        print(f"Época {epoch}/{epochs} | Pérdida Promedio: {avg_loss:.4f}")

    # 4. Guardar el modelo entrenado
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/vae_mnist_2d.pth")
    print("¡Entrenamiento completado y modelo guardado!")

if __name__ == "__main__":
    train()