import sys
import json
import argparse
import subprocess
from pathlib import Path
from contextlib import nullcontext

import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from dataset import make_dataset, make_dataloader
from model import ChessTransformer
from config import load_config 

def get_device() -> tuple[torch.device, bool]:
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(True)
        return torch.device("cuda"), True

    if torch.backends.mps.is_available():
        return torch.device("mps"), False

    return torch.device("cpu"), False


def unwrap(model: nn.Module) -> nn.Module:
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def gpu_stats():
    result = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )

    util, used, total = result.decode().strip().split(",")

    return {
        "gpu_util": int(util),
        "mem_used_mb": int(used),
        "mem_total_mb": int(total),
    }


def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    use_amp,
    epoch,
    gpu_history,
) -> float:
    model.train()
    criterion = nn.L1Loss()

    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch:02d}", leave=True)

    for batch in pbar:
        bt = batch["board_tensors"].to(device, non_blocking=True, dtype = torch.float32)
        tf = batch["time_features"].to(device, non_blocking=True)
        msk = batch["attention_mask"].to(device, non_blocking=True)

        # New shard format stores both values together: [B, 2].
        elo = batch["elos"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with (
            torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            )
            if use_amp
            else nullcontext()
        ):
            # Keep this unchanged if your model currently expects:
            # model(board_tensors, time_features, attention_mask)
            pred = model(bt, tf, msk)
            print(
                "INPUT DTYPES:",
                "bt=", bt.dtype,
                "tf=", tf.dtype,
                "msk=", msk.dtype,
                "elo=", elo.dtype,
            )
            loss = criterion(pred, elo)

        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if use_amp and n_batches % 100 == 0:
            stats = gpu_stats()

            gpu_history.append(
                {
                    "epoch": epoch,
                    "batch": n_batches,
                    **stats,
                }
            )

        pbar.set_postfix(
            {
                "loss": f"{total_loss / n_batches:.1f}",
            }
        )

    return total_loss / max(n_batches, 1)


@torch.inference_mode()
def evaluate(
    model,
    loader,
    device,
    use_amp,
) -> float:
    model.eval()

    total_mae = 0.0
    n_games = 0

    for batch in tqdm(loader, desc="Eval", leave=False):
        bt = batch["board_tensors"].to(device, non_blocking=True, dtype=torch.float32)
        tf = batch["time_features"].to(device, non_blocking=True)
        msk = batch["attention_mask"].to(device, non_blocking=True)

        # Shape: [B, 2], columns are [white_elo, black_elo].
        elo = batch["elos"].to(device, non_blocking=True)

        with (
            torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            )
            if use_amp
            else nullcontext()
        ):
            pred = model(bt, tf, msk)

        # Per-game mean absolute error across white and black Elo.
        total_mae += torch.abs(pred - elo).mean(dim=1).sum().item()
        n_games += elo.shape[0]

    return total_mae / max(n_games, 1)


def run(cfg):
    training_store = cfg.training_store
    models_dir = cfg.models_dir
    epochs = cfg.epochs
    batch_size = cfg.batch_size
    lr = cfg.learning_rate
    d_model = cfg.d_model
    n_heads = cfg.n_heads
    n_layers = cfg.n_layers
    num_workers = cfg.num_workers
    shuffle_buffer = cfg.shuffle_buffer
    
    device, use_amp = get_device()

    scaler = (
        torch.amp.GradScaler("cuda")
        if use_amp
        else None
    )

    print(f"Device: {device} | AMP: {use_amp}")

    # These are optional explicit dataset objects. They are useful for a
    # quick construction/smoke test, while loaders below create their own.
    train_dataset = make_dataset(training_store, "train")
    test_dataset = make_dataset(training_store, "test")
    del train_dataset, test_dataset

    # Do not use DataLoader(..., shuffle=True) here:
    # WebDataset handles stream/shard/buffer shuffle internally for train.
    train_loader = make_dataloader(
        training_store,
        "train",
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle_buffer=shuffle_buffer
    )

    test_loader = make_dataloader(
        training_store,
        "test",
        batch_size=batch_size * 2,
        num_workers=num_workers,
    )

    model = ChessTransformer(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_model * 4,
    ).to(device)

    if use_amp:
        model = torch.compile(model)

    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
    )

    Path(models_dir).mkdir(parents=True, exist_ok=True)

    best = float("inf")
    history = []
    gpu_history = []

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            use_amp,
            epoch,
            gpu_history,
        )

        mae = evaluate(
            model,
            test_loader,
            device,
            use_amp,
        )

        scheduler.step()

        history.append(
            {
                "epoch": epoch,
                "loss": loss,
                "elo_mae": mae,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        print(
            f"Epoch {epoch:02d} | "
            f"loss {loss:.1f} | "
            f"elo_mae {mae:.1f}"
        )

        if mae < best:
            best = mae

            torch.save(
                unwrap(model).state_dict(),
                f"{models_dir}/transformer_best.pt",
            )

            print(f"  ✓ best saved (elo_mae={best:.1f})")

    torch.save(
        unwrap(model).state_dict(),
        f"{models_dir}/transformer_final.pt",
    )

    with open(f"{models_dir}/transformer_history.json", "w") as f:
        json.dump(history, f, indent=2)

    with open(f"{models_dir}/gpu_stats.json", "w") as f:
        json.dump(gpu_history, f, indent=2)

    print(f"\nDone. Best elo_mae: {best:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/train.toml",
        help="Path to training TOML configuration",
    )

    args = parser.parse_args()
    config = load_config(args.config)

    run(config)