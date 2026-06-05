import matplotlib.pyplot as plt
import json
import os


def plot_comparison():
    """
    Generate comparison plots of AEVB vs Wake-Sleep training efficiency.
    Replicates Figure 2 in the Kingma & Welling (2013) paper.
    """
    results_dir = "results/metrics"

    # All the configurations we implemented and saved previously
    mnist_configs = [
        {"latent_dim": 3, "dataset": "mnist"},
        {"latent_dim": 5, "dataset": "mnist"},
        {"latent_dim": 10, "dataset": "mnist"},
        {"latent_dim": 20, "dataset": "mnist"},
        {"latent_dim": 200, "dataset": "mnist"},
    ]

    frey_configs = [
        {"latent_dim": 2, "dataset": "frey_face"},
        {"latent_dim": 5, "dataset": "frey_face"},
        {"latent_dim": 10, "dataset": "frey_face"},
        {"latent_dim": 20, "dataset": "frey_face"},
    ]

    # Create a 2x5 grid
    fig = plt.figure(figsize=(16, 7))
    
    stride = 100

    # ==========================================
    # 1. MNIST PLOTS (Top Row: Subplots 1 to 5)
    # ==========================================
    for idx, config in enumerate(mnist_configs):
        ax = plt.subplot(2, 5, idx + 1)
        latent_dim = config["latent_dim"]
        dataset = config["dataset"]

        aevb_path = os.path.join(results_dir, f"aevb_{dataset}_{latent_dim}d_metrics.json")
        ws_path = os.path.join(results_dir, f"wake_sleep_{dataset}_{latent_dim}d_metrics.json")

        data_found = False
        if os.path.exists(aevb_path) and os.path.exists(ws_path):
            with open(aevb_path) as f:
                aevb_metrics = json.load(f)
            with open(ws_path) as f:
                ws_metrics = json.load(f)
            data_found = True

            ax.semilogx(
                aevb_metrics["samples_evaluated"][::stride],
                [-x for x in aevb_metrics["train_loss"]][::stride],
                "r-", linewidth=2
            )
            ax.semilogx(
                aevb_metrics["samples_evaluated"][::stride],
                [-x for x in aevb_metrics["test_loss"]][::stride],
                "r--", linewidth=2
            )
            ax.semilogx(
                ws_metrics["samples_evaluated"][::stride],
                [-x for x in ws_metrics["train_loss"]][::stride],
                "g-", linewidth=2
            )
            ax.semilogx(
                ws_metrics["samples_evaluated"][::stride],
                [-x for x in ws_metrics["test_loss"]][::stride],
                "g-.", linewidth=2
            )

        ax.set_xlabel("# Training samples evaluated")
        ax.set_ylabel(r"$\mathcal{L}$")
        ax.set_title(f"MNIST, $N_z$={latent_dim}")
        ax.grid(True, alpha=0.3)

        if not data_found:
            ax.text(0.5, 0.5, f"No data for Nz={latent_dim}", ha="center", va="center", transform=ax.transAxes)

    # ==========================================
    # 2. FREY FACE PLOTS (Bottom Row: Subplots 7 to 10)
    # ==========================================
    for idx, config in enumerate(frey_configs):
        ax = plt.subplot(2, 5, idx + 7)
        latent_dim = config["latent_dim"]
        dataset = config["dataset"]

        aevb_path = os.path.join(results_dir, f"aevb_{dataset}_{latent_dim}d_metrics.json")
        ws_path = os.path.join(results_dir, f"wake_sleep_{dataset}_{latent_dim}d_metrics.json")

        data_found = False
        if os.path.exists(aevb_path) and os.path.exists(ws_path):
            with open(aevb_path) as f:
                aevb_metrics = json.load(f)
            with open(ws_path) as f:
                ws_metrics = json.load(f)
            data_found = True

            ax.semilogx(
                aevb_metrics["samples_evaluated"][::stride],
                [-x for x in aevb_metrics["train_loss"]][::stride],
                "r-", linewidth=2
            )
            ax.semilogx(
                aevb_metrics["samples_evaluated"][::stride],
                [-x for x in aevb_metrics["test_loss"]][::stride],
                "r--", linewidth=2
            )
            ax.semilogx(
                ws_metrics["samples_evaluated"][::stride],
                [-x for x in ws_metrics["train_loss"]][::stride],
                "g-", linewidth=2
            )
            ax.semilogx(
                ws_metrics["samples_evaluated"][::stride],
                [-x for x in ws_metrics["test_loss"]][::stride],
                "g-.", linewidth=2
            )

        ax.set_xlabel("# Training samples evaluated")
        ax.set_ylabel(r"$\mathcal{L}$")
        ax.set_title(f"Frey Face, $N_z$={latent_dim}")
        ax.grid(True, alpha=0.3)

        if not data_found:
            ax.text(0.5, 0.5, f"No data for Nz={latent_dim}", ha="center", va="center", transform=ax.transAxes)

    # ==========================================
    # 3. GLOBAL AXIS CROPPING
    # ==========================================
    for ax in fig.axes:
        ax.set_xlim([1e5, 1e8]) 

        title = ax.get_title()
        if 'MNIST' in title:
            ax.set_ylim([-150, -70])
        elif 'Frey Face' in title:
            ax.set_ylim([0, 1600])

    ax_legend = plt.subplot(2, 5, 6)
    ax_legend.axis("off")
            
    # Custom handles to match the plotted styles perfectly
    handles = [
        plt.Line2D([0], [0], color="g", linestyle="-.", linewidth=2),
        plt.Line2D([0], [0], color="g", linestyle="-", linewidth=2),
        plt.Line2D([0], [0], color="r", linestyle="--", linewidth=2),
        plt.Line2D([0], [0], color="r", linestyle="-", linewidth=2),
    ]
    labels = ["Wake-Sleep (test)", "Wake-Sleep (train)", "AEVB (test)", "AEVB (train)"]

    ax_legend.legend(handles, labels, loc="center", fontsize=11, frameon=True, edgecolor="black")

    plt.tight_layout()
    plt.savefig("results/comparison_aevb_vs_wake_sleep.png", dpi=150, bbox_inches="tight")
    print("Comparison plot saved to results/comparison_aevb_vs_wake_sleep.png")


if __name__ == "__main__":
    plot_comparison()
