import sys
import json
import argparse
import subprocess  # GPU LOG
from pathlib import Path
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from dataset import ChessGameDataset, collate_fn
from model import ChessTransformer


def get_device() -> tuple[torch.device, bool]:
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(True)
        return torch.device("cuda"), True

    if torch.backends.mps.is_available():
        return torch.device("mps"), False

    return torch.device("cpu"), False


def unwrap(model: nn.Module) -> nn.Module:
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def gpu_stats():  # GPU LOG
    result = subprocess.check_output(  # GPU LOG
        [
            "nvidia-smi",  # GPU LOG
            "--query-gpu=utilization.gpu,memory.used,memory.total",  # GPU LOG
            "--format=csv,noheader,nounits",  # GPU LOG
        ]
    )  # GPU LOG

    util, used, total = result.decode().strip().split(",")  # GPU LOG

    return {  # GPU LOG
        "gpu_util": int(util),  # GPU LOG
        "mem_used_mb": int(used),  # GPU LOG
        "mem_total_mb": int(total),  # GPU LOG
    }


def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    use_amp,
    epoch,
    gpu_history,  # GPU LOG
) -> float:

    model.train()
    criterion = nn.L1Loss()

    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(
        loader,
        desc=f"Epoch {epoch:02d}",
        leave=True
    )

    for batch in pbar:

        bt = batch["board_tensors"].to(
            device,
            non_blocking=True
        )

        tf = batch["time_features"].to(
            device,
            non_blocking=True
        )

        msk = batch["attention_mask"].to(
            device,
            non_blocking=True
        )

        elo = torch.stack(
            [
                batch["white_elo"].to(device, non_blocking=True),
                batch["black_elo"].to(device, non_blocking=True),
            ],
            dim=1,
        )

        optimizer.zero_grad(set_to_none=True)

        with (
            torch.autocast(
                device_type="cuda",
                dtype=torch.float16
            )
            if use_amp
            else nullcontext()
        ):
            pred = model(bt, tf, msk)
            loss = criterion(pred, elo)

        if scaler:

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)

            nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            scaler.step(optimizer)

            scaler.update()

        else:

            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            optimizer.step()


        total_loss += loss.item()

        n_batches += 1


        if n_batches % 100 == 0:  # GPU LOG
            stats = gpu_stats()  # GPU LOG

            gpu_history.append({  # GPU LOG
                "epoch": epoch,  # GPU LOG
                "batch": n_batches,  # GPU LOG
                **stats,  # GPU LOG
            })


        pbar.set_postfix(
            {
                "loss": f"{total_loss / n_batches:.1f}"
            }
        )

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    use_amp
) -> float:

    model.eval()

    total_mae = 0.0
    n = 0

    for batch in tqdm(
        loader,
        desc="Eval",
        leave=False
    ):

        bt = batch["board_tensors"].to(
            device,
            non_blocking=True
        )

        tf = batch["time_features"].to(
            device,
            non_blocking=True
        )

        msk = batch["attention_mask"].to(
            device,
            non_blocking=True
        )

        we = batch["white_elo"].to(
            device,
            non_blocking=True
        )

        be = batch["black_elo"].to(
            device,
            non_blocking=True
        )


        with (
            torch.autocast(
                device_type="cuda",
                dtype=torch.float16
            )
            if use_amp
            else nullcontext()
        ):
            pred = model(bt, tf, msk)


        total_mae += (
            (
                torch.abs(pred[:, 0] - we)
                +
                torch.abs(pred[:, 1] - be)
            )
            / 2
        ).sum().item()

        n += len(we)


    return total_mae / max(n, 1)



def run(
    features_path,
    models_dir,
    epochs,
    batch_size,
    lr,
    d_model,
    n_heads,
    n_layers
):

    device, use_amp = get_device()

    scaler = (
        torch.amp.GradScaler("cuda")
        if use_amp
        else None
    )

    print(
        f"Device: {device} | AMP: {use_amp}"
    )


    train_ds = ChessGameDataset(
        features_path,
        split="train"
    )

    test_ds = ChessGameDataset(
        features_path,
        split="test"
    )


    loader_kw = dict(
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=use_amp
    )


    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        **loader_kw
    )


    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        **loader_kw
    )


    model = ChessTransformer(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_model * 4,
    ).to(device)


    if use_amp:
        model = torch.compile(model)


    print(
        f"Params: {sum(p.numel() for p in model.parameters()):,}"
    )


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4
    )


    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs
    )


    Path(models_dir).mkdir(
        parents=True,
        exist_ok=True
    )


    best = float("inf")
    history = []
    gpu_history = []  # GPU LOG


    for epoch in range(
        1,
        epochs + 1
    ):

        loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            use_amp,
            epoch,
            gpu_history,  # GPU LOG
        )


        mae = evaluate(
            model,
            test_loader,
            device,
            use_amp
        )


        scheduler.step()


        history.append(
            {
                "epoch": epoch,
                "loss": loss,
                "elo_mae": mae,
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
                f"{models_dir}/transformer_best.pt"
            )

            print(
                f"  ✓ best saved (elo_mae={best:.1f})"
            )


    torch.save(
        unwrap(model).state_dict(),
        f"{models_dir}/transformer_final.pt"
    )


    with open(
        f"{models_dir}/transformer_history.json",
        "w"
    ) as f:
        json.dump(
            history,
            f,
            indent=2
        )


    with open(
        f"{models_dir}/gpu_stats.json",
        "w"
    ) as f:  # GPU LOG
        json.dump(
            gpu_history,
            f,
            indent=2
        )


    print(
        f"\nDone. Best elo_mae: {best:.1f}"
    )



if __name__ == "__main__":

    p = argparse.ArgumentParser()

    p.add_argument(
        "--features-path",
        default="data/features/features.parquet"
    )

    p.add_argument(
        "--models-dir",
        default="models"
    )

    p.add_argument(
        "--epochs",
        type=int,
        default=10
    )

    p.add_argument(
        "--batch-size",
        type=int,
        default=32
    )

    p.add_argument(
        "--lr",
        type=float,
        default=1e-4
    )

    p.add_argument(
        "--d-model",
        type=int,
        default=256
    )

    p.add_argument(
        "--n-heads",
        type=int,
        default=4
    )

    p.add_argument(
        "--n-layers",
        type=int,
        default=4
    )


    args = p.parse_args()


    run(
        args.features_path,
        args.models_dir,
        args.epochs,
        args.batch_size,
        args.lr,
        args.d_model,
        args.n_heads,
        args.n_layers,
    )