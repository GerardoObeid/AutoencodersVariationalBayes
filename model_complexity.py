import os
import json
import numpy as np
import matplotlib.pyplot as plt
from models.vae import VAE

def extract_and_plot_complexity():
    latent_dims = [2, 3, 5, 10, 20, 200]
    input_dim = 560  # Frey Face input dimension
    results_dir = "results/metrics"
    
    # Storage for table and plot
    param_counts = []
    final_train_elbo = []
    final_test_elbo = []
    table_rows = []
    
    for nz in latent_dims:
        # 1. Calculate exact parameters
        model = VAE(input_dim=input_dim, latent_dim=nz, dataset='mnist')
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        param_counts.append(total_params)
        
        # 2. Extract final converged ELBO scores from your JSON runs
        aevb_path = os.path.join(results_dir, f"aevb_mnist_{nz}d_metrics.json")

        if os.path.exists(aevb_path):
            with open(aevb_path) as f:
                metrics = json.load(f)
            # Take the final epoch value (Kingma & Welling lower bound L)

            train_score = -metrics["train_loss"][-1] if metrics["train_loss"][-1] > 0 else metrics["train_loss"][-1]
            test_score = -metrics["test_loss"][-1] if metrics["test_loss"][-1] > 0 else metrics["test_loss"][-1]
        else:
            # Fallbacks if you haven't finished running all dimensions yet
            train_score = np.nan
            test_score = np.nan
            
        final_train_elbo.append(train_score)
        final_test_elbo.append(test_score)
        
        # Format row for markdown output
        table_rows.append(f"| {nz} | {total_params:,} | {train_score:.2f} | {test_score:.2f} |")

    # ==========================================
    #  PRINT Statistics
    # ==========================================
    print("\n### Model Complexity vs. Performance Table\n")
    print("| Latent Dim ($N_z$) | Total Trainable Parameters | Final Train ELBO ($\mathcal{L}$) | Final Test ELBO ($\mathcal{L}$) |")
    print("|--------------------|----------------------------|----------------------------------|---------------------------------|")
    for row in table_rows:
        print(row)

    # ==========================================
    #  Plot Statistics
    # ==========================================
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # X-axis setup (Using a categorical spacing so 200 doesn't squish 2, 3, 5)
    x_indices = np.arange(len(latent_dims))

    # Left Y-Axis: ELBO Scores
    color = 'tab:red'
    ax1.set_xlabel('Latent Dimensionality ($N_z$)', fontweight='bold', labelpad=10)
    ax1.set_ylabel('Converged ELBO ($\mathcal{L}$)', color=color, fontweight='bold')
    line1 = ax1.plot(x_indices, final_train_elbo, color='red', marker='o', linewidth=2, label='Train ELBO (Capacity)')
    line2 = ax1.plot(x_indices, final_test_elbo, color='red', linestyle='--', marker='x', linewidth=2, label='Test ELBO (Generalization)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    # Right Y-Axis: Parameter Count (Log scale because 200d blows up parameter count)
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Total Model Parameters (Log Scale)', color=color, fontweight='bold')
    line3 = ax2.plot(x_indices, param_counts, color='blue', linestyle='-.', marker='s', linewidth=2, label='Parameter Count')
    ax2.set_yscale('log')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Set pristine X-axis ticks
    plt.xticks(x_indices, [str(nz) for nz in latent_dims])
    
    # Consolidated Legend
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=True, edgecolor='black')
    
    plt.title('The Complexity Trade-off: Parameter Growth vs. Overfitting', fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/complexity_vs_overfitting.png", dpi=300, bbox_inches='tight')
    print("\nComplexity analysis plot saved to results/complexity_vs_overfitting.png")

if __name__ == "__main__":
    extract_and_plot_complexity()