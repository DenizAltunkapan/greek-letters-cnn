import os
import random
import argparse
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch

from config_utils import default_config_path, load_config, resolve_ml_path
from dataset import get_dataloaders
from model import GreekLetterCNN
from trainer import (
    compute_confusion,
    format_per_class_report,
    per_class_metrics,
    top_confusions,
    train_model,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def config_label(cfg):
    return (
        f"{cfg['channels']} | h={cfg['hidden_size']} "
        f"| d={cfg['dropout']} | lr={cfg['lr']} | wd={cfg['weight_decay']}"
    )


def aggregate_curves(curves):
    max_len = max(len(curve) for curve in curves)
    matrix = np.full((len(curves), max_len), np.nan, dtype=np.float32)

    for idx, curve in enumerate(curves):
        matrix[idx, : len(curve)] = curve

    mean_curve = np.nanmean(matrix, axis=0)
    std_curve = np.nanstd(matrix, axis=0)
    return mean_curve, std_curve


def ensure_parent_dir(file_path):
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def add_timestamp_suffix(file_path, timestamp):
    root, ext = os.path.splitext(file_path)
    return f"{root}_{timestamp}{ext}"


def main():
    parser = argparse.ArgumentParser(description="Train and compare CNN architectures.")
    parser.add_argument(
        "--config",
        type=str,
        default=default_config_path(),
        help="Path to shared YAML config file.",
    )
    args = parser.parse_args()

    loaded_config, selected_config_path = load_config(args.config)

    paths_cfg = loaded_config.get("paths", {})
    data_cfg = loaded_config.get("data", {})
    experiment_cfg = loaded_config.get("experiment", {})
    trainer_cfg = loaded_config.get("trainer", {})
    configs = loaded_config.get("search_space", [])

    stable_model_path = resolve_ml_path(paths_cfg.get("model_path", "models/best_model.pth"))
    stable_plot_path = resolve_ml_path(paths_cfg.get("plot_path", "plots/architecture_comparison.png"))
    data_dir = resolve_ml_path(data_cfg.get("data_dir", "data"))
    batch_size = int(data_cfg.get("batch_size", 32))
    seeds = experiment_cfg.get("seeds", [42, 1337, 2026])
    epochs = int(experiment_cfg.get("epochs", 30))

    patience = int(trainer_cfg.get("patience", 6))
    min_delta = float(trainer_cfg.get("min_delta", 1e-4))
    scheduler_patience = int(trainer_cfg.get("scheduler_patience", 3))
    scheduler_factor = float(trainer_cfg.get("scheduler_factor", 0.5))

    if not configs:
        raise ValueError("Config file contains no entries in 'search_space'.")

    train_loader, val_loader, _, classes = get_dataloaders(data_dir, batch_size)
    num_classes = len(classes)

    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    timestamped_model_path = add_timestamp_suffix(stable_model_path, run_timestamp)
    timestamped_plot_path = add_timestamp_suffix(stable_plot_path, run_timestamp)

    ensure_parent_dir(stable_model_path)
    ensure_parent_dir(stable_plot_path)
    ensure_parent_dir(timestamped_model_path)
    ensure_parent_dir(timestamped_plot_path)

    print(f"Device: {device}")
    print(f"Config file: {selected_config_path}")
    print(f"Timestamp: {run_timestamp}")
    print(f"Seeds: {seeds}")

    curve_data = []

    best_config_state = None
    best_config = None
    best_seed = None
    best_mean_score = -1.0

    for index, cfg in enumerate(configs, start=1):
        print(f"\nEvaluating config {index}/{len(configs)}: {cfg}")

        seed_metrics = []
        val_curves = []
        train_curves = []

        config_best_run_score = -1.0
        config_best_run_state = None
        config_best_seed = None

        for seed in seeds:
            print(f"\n  Seed {seed}")
            set_seed(seed)

            model = GreekLetterCNN(
                num_classes=num_classes,
                channels=cfg["channels"],
                hidden_size=cfg["hidden_size"],
                dropout=cfg["dropout"],
            ).to(device)

            train_result = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                epochs=epochs,
                lr=cfg["lr"],
                weight_decay=cfg["weight_decay"],
                patience=patience,
                min_delta=min_delta,
                scheduler_patience=scheduler_patience,
                scheduler_factor=scheduler_factor,
            )

            run_score = float(train_result["best_val_acc"])
            val_curves.append(train_result["val_acc_history"])
            train_curves.append(train_result["train_loss_history"])
            seed_metrics.append(
                {
                    "seed": seed,
                    "best_val_acc": run_score,
                    "best_epoch": int(train_result["best_epoch"]),
                    "epochs_trained": int(train_result["epochs_trained"]),
                }
            )

            if run_score > config_best_run_score:
                config_best_run_score = run_score
                config_best_run_state = train_result["best_state"]
                config_best_seed = seed

        scores = [entry["best_val_acc"] for entry in seed_metrics]
        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores))

        print(
            f"Config summary: mean best val acc = {mean_score:.4f} ± {std_score:.4f} "
            f"(best seed: {config_best_seed}, best single run: {config_best_run_score:.4f})"
        )

        curve_data.append({"config": cfg, "val_curves": val_curves, "train_curves": train_curves})

        if mean_score > best_mean_score:
            best_mean_score = mean_score
            best_config = cfg
            best_seed = config_best_seed
            best_config_state = config_best_run_state

    plt.figure(figsize=(12, 7))
    for curve_entry in curve_data:
        mean_curve, std_curve = aggregate_curves(curve_entry["val_curves"])
        epochs_axis = np.arange(1, len(mean_curve) + 1)

        plt.plot(epochs_axis, mean_curve, label=config_label(curve_entry["config"]))
        plt.fill_between(
            epochs_axis,
            mean_curve - std_curve,
            mean_curve + std_curve,
            alpha=0.15,
        )

    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("Architecture Comparison (mean ± std across seeds)")
    plt.legend(fontsize=8)
    plt.tight_layout()

    plt.savefig(timestamped_plot_path, dpi=150)
    plt.savefig(stable_plot_path, dpi=150)
    plt.close()

    model_payload = {
        "model_state": best_config_state,
        "config": {
            "channels": best_config["channels"],
            "hidden_size": best_config["hidden_size"],
            "dropout": best_config["dropout"],
        },
        "selection_metric": "mean_best_val_acc_across_seeds",
        "selection_score": float(best_mean_score),
        "selected_seed": best_seed,
        "seeds": seeds,
        "classes": classes,
    }
    torch.save(model_payload, timestamped_model_path)
    torch.save(model_payload, stable_model_path)

    print("\nBest config (by mean validation score across seeds):", best_config)
    print(f"Best mean validation score: {best_mean_score:.4f}")
    print(f"Selected seed for saved checkpoint: {best_seed}")
    print(f"Comparison plot saved to {timestamped_plot_path}")
    print(f"Latest plot also saved to {stable_plot_path}")
    print(f"Best model saved to {timestamped_model_path}")
    print(f"Latest model also saved to {stable_model_path}")

    best_model = GreekLetterCNN(
        num_classes=num_classes,
        channels=best_config["channels"],
        hidden_size=best_config["hidden_size"],
        dropout=best_config["dropout"],
    ).to(device)
    best_model.load_state_dict(best_config_state)

    val_confusion = compute_confusion(best_model, val_loader, num_classes, device)
    val_rows = per_class_metrics(val_confusion, classes)
    print()
    print(format_per_class_report(val_rows, title="Per-class validation performance (best model, worst first)"))
    print("\nMost frequent validation confusions (true -> predicted):")
    for true_class, pred_class, count in top_confusions(val_confusion, classes, top_k=10):
        print(f"  {true_class:<10} -> {pred_class:<10} {count}")


if __name__ == "__main__":
    main()
