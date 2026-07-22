"""
train_transformer.py — Training loop for Elo prediction transformer.

Transformer predicts white_elo + black_elo from game sequence.
Move quality annotation is handled by Stockfish separately.

Usage:
  uv run src/training/train_transformer.py
  uv run src/training/train_transformer.py --epochs 20 --batch-size 32
"""

import sys
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from dataset import ChessGameDataset, collate_fn
from model   import ChessTransformer


def train_one_epoch(
    model:     ChessTransformer,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    device:    torch.device,
    epoch:     int,
) -> dict:
    model.train()
    criterion  = nn.L1Loss()
    total_loss = 0.0
    n_batches  = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch:02d}", leave=True)

    for batch in pbar:
        board_tensors  = batch["board_tensors"].to(device)
        time_features  = batch["time_features"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        white_elo      = batch["white_elo"].to(device)
        black_elo      = batch["black_elo"].to(device)

        elo_pred = model(board_tensors, time_features, attention_mask)

        elo_true = torch.stack([white_elo, black_elo], dim=1)  # (B, 2)
        loss     = criterion(elo_pred, elo_true)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

        pbar.set_postfix({"elo_loss": f"{total_loss / n_batches:.1f}"})

    return {"elo_loss": total_loss / n_batches}


def evaluate(
    model:  ChessTransformer,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()

    total_elo_mae = 0.0
    n_games       = 0

    pbar = tqdm(loader, desc="Evaluating", leave=False)

    with torch.no_grad():
        for batch in pbar:
            board_tensors  = batch["board_tensors"].to(device)
            time_features  = batch["time_features"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            white_elo      = batch["white_elo"].to(device)
            black_elo      = batch["black_elo"].to(device)

            elo_pred = model(board_tensors, time_features, attention_mask)

            white_mae = torch.abs(elo_pred[:, 0] - white_elo)
            black_mae = torch.abs(elo_pred[:, 1] - black_elo)
            elo_mae   = ((white_mae + black_mae) / 2).sum()

            total_elo_mae += elo_mae.item()
            n_games       += len(white_elo)

    return {"elo_mae": total_elo_mae / max(n_games, 1)}


def run(
    features_path: str,
    models_dir:    str,
    epochs:        int,
    batch_size:    int,
    lr:            float,
    d_model:       int,
    n_heads:       int,
    n_layers:      int,
):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    print("\nLoading datasets ...")
    train_ds = ChessGameDataset(features_path, split="train")
    test_ds  = ChessGameDataset(features_path, split="test")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
        persistent_workers=True,
        prefetch_factor=2,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        persistent_workers=True,
        prefetch_factor=2,
    )
    model = ChessTransformer(
        d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, d_ff=d_model * 4,
    ).to(device)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    Path(models_dir).mkdir(parents=True, exist_ok=True)
    best_elo_mae = float("inf")
    history      = []

    print(f"\nTraining for {epochs} epochs ...")
    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, epoch)
        eval_metrics  = evaluate(model, test_loader, device)
        scheduler.step()

        row = {
            "epoch": epoch,
            "lr":    scheduler.get_last_lr()[0],
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}":   v for k, v in eval_metrics.items()},
        }
        history.append(row)

        print(f"Epoch {epoch:02d} | elo_mae {eval_metrics['elo_mae']:.1f}")

        if eval_metrics["elo_mae"] < best_elo_mae:
            best_elo_mae = eval_metrics["elo_mae"]
            torch.save(model.state_dict(), f"{models_dir}/transformer_best.pt")
            print(f"  ✓ best model saved (elo_mae={best_elo_mae:.1f})")

    torch.save(model.state_dict(), f"{models_dir}/transformer_final.pt")
    with open(f"{models_dir}/transformer_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nDone. Best elo_mae: {best_elo_mae:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chess transformer — Elo prediction")
    parser.add_argument("--features-path", type=str,   default="data/features/features.parquet")
    parser.add_argument("--models-dir",    type=str,   default="models")
    parser.add_argument("--epochs",        type=int,   default=10)
    parser.add_argument("--batch-size",    type=int,   default=32)
    parser.add_argument("--lr",            type=float, default=1e-4)
    parser.add_argument("--d-model",       type=int,   default=256)
    parser.add_argument("--n-heads",       type=int,   default=4)
    parser.add_argument("--n-layers",      type=int,   default=4)
    args = parser.parse_args()

    run(
        features_path=args.features_path,
        models_dir=args.models_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
    )