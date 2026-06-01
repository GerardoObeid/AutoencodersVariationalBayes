#!/usr/bin/env python3
"""
Parallel training script for all VAE configurations.
Runs multiple experiments concurrently on GPU.
"""

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import time


def train_model(task):
    """Train a single model configuration (either AEVB or Wake-Sleep)."""
    dataset = task["dataset"]
    latent_dim = task["latent_dim"]
    algo = task["run_algo"]

    if dataset == "frey_face":
        epochs = "6107"
    else:               
        epochs = "200"

    try:
        if algo == "aevb":
            script = "train_aevb_metrics.py"
            name = "AEVB"
        elif algo == "wake_sleep":
            script = "train_wake_sleep_metrics.py"
            name = "Wake-Sleep"
        else:
            return False

        print(f"[{dataset} Nz={latent_dim}] Starting {name} training...")
        cmd = [
            sys.executable,
            script,
            "--dataset", dataset,
            "--latent_dim", str(latent_dim),
            "--epochs", epochs,
            "--batch_size", "100",
            "--lr", "0.02",
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"[{dataset} Nz={latent_dim}] {name} FAILED")
            print(result.stderr)
            return False
        else:
            print(f"[{dataset} Nz={latent_dim}] {name} COMPLETED")
            return True

    except Exception as e:
        print(f"[{dataset} Nz={latent_dim}] ERROR running {name}: {e}")
        return False


def main():
    configs = [
        # MNIST experiments
        {"dataset": "mnist", "latent_dim": 3, "algorithm": "both"},
        {"dataset": "mnist", "latent_dim": 5, "algorithm": "both"},
        {"dataset": "mnist", "latent_dim": 10, "algorithm": "both"},
        {"dataset": "mnist", "latent_dim": 20, "algorithm": "both"},
        {"dataset": "mnist", "latent_dim": 200, "algorithm": "both"},
        # Frey Face experiments
        {"dataset": "frey_face", "latent_dim": 2, "algorithm": "both"},
        {"dataset": "frey_face", "latent_dim": 5, "algorithm": "both"},
        {"dataset": "frey_face", "latent_dim": 10, "algorithm": "both"},
        {"dataset": "frey_face", "latent_dim": 20, "algorithm": "both"},
    ]

    # Flatten configs into individual executable tasks to run both algorithms perfectly in parallel
    tasks = []
    for config in configs:
        if config["algorithm"] in ["aevb", "both"]:
            tasks.append({
                "dataset": config["dataset"], 
                "latent_dim": config["latent_dim"], 
                "run_algo": "aevb"
            })
        if config["algorithm"] in ["wake_sleep", "both"]:
            tasks.append({
                "dataset": config["dataset"], 
                "latent_dim": config["latent_dim"], 
                "run_algo": "wake_sleep"
            })

    print("=" * 70)
    print("PARALLEL VAE TRAINING")
    print("=" * 70)
    print(f"Total base configs: {len(configs)}")
    print(f"Total concurrent tasks to run: {len(tasks)}")
    print("=" * 70)

    start_time = time.time()

    # Controls maximum parallel processes on your GPU.
    max_workers = 9

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(train_model, task): task for task in tasks}

        completed = 0
        failed = 0

        for future in as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
                algo_name = "AEVB" if task["run_algo"] == "aevb" else "Wake-Sleep"
                if result:
                    completed += 1
                    print(f"\n✓ COMPLETED: {algo_name} | {task['dataset']} Nz={task['latent_dim']}")
                else:
                    failed += 1
                    print(f"\n✗ FAILED: {algo_name} | {task['dataset']} Nz={task['latent_dim']}")
            except Exception as e:
                failed += 1
                algo_name = "AEVB" if task["run_algo"] == "aevb" else "Wake-Sleep"
                print(f"\n✗ ERROR: {algo_name} | {task['dataset']} Nz={task['latent_dim']}: {e}")

    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"Completed tasks: {completed}/{len(tasks)}")
    print(f"Failed tasks: {failed}/{len(tasks)}")
    print(f"Total time: {elapsed/3600:.2f} hours")
    print("=" * 70)

    if failed == 0:
        print("\nGenerating comparison plot...")
        result = subprocess.run([sys.executable, "plot_comparison.py"], capture_output=False)
        if result.returncode == 0:
            print("✓ Comparison plot generated!")
        else:
            print("✗ Failed to generate comparison plot")
            return False

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)