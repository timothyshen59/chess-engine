

import polars as pl 
import numpy as np 
import xgboost as xgb 
import argparse 
import json 
from pathlib import Path 

FEATURE_COLS = [ 
    "num_legal_moves", 
    "material_balance", 
    "king_safety_score", 
    "time_pressure_flag", 
    "avg_time_spent",           
]

LABEL_COL = "centipawn_loss"

def to_label(cp: float) -> int:
    if cp < 10:  return 0  # good
    if cp < 25:  return 1  # inaccuracy
    if cp < 100: return 2  # mistake
    return 3               # blunder

LABEL_NAMES = {0: "good", 1: "inaccuracy", 2: "mistake", 3: "blunder"}


def load_data(features_dir: str, labels_dir: str) -> pl.DataFrame: 
    """
    Load features + labels and join on game_id + move_number + color
    """
    
    print("Loading features...")
    features = pl.read_parquet(f"{features_dir}/features.parquet")
    print(f" {len(features):,} rows")
    
    print("Loading labels...")
    labels = pl.read_parquet(f"{labels_dir}/labels.parquet")
    print(f" {len(labels):,} rows")
    
    df = features.join(labels, on=["game_id", "move_number", "color"], how="inner")
    print(f"  {len(df):,} rows after join")

    needed = FEATURE_COLS + [LABEL_COL, "game_id"]
    df = df.drop_nulls(subset=needed)
    print(f"  {len(df):,} rows after dropping nulls")
    
    df = df.with_columns(
        pl.col("time_pressure_flag").cast(pl.Int8)
    )
 
    return df


def split_by_game(df: pl.DataFrame, test_fraction: float = 0.2) -> tuple:
    """
    Split into train/test by game_id, not by row.
 
    Why: if you split by row, moves from the same game appear in both
    train and test. The model memorizes game-level patterns and appears
    to generalize when it's actually interpolating within a game.
    Splitting by game_id ensures test games are completely unseen.
 
    seed=42: makes the split reproducible across runs so metrics are comparable.
    """
    game_ids   = df["game_id"].unique().to_list()
    n_test     = int(len(game_ids) * test_fraction)
 
    rng        = np.random.default_rng(seed=42)
    shuffled   = rng.permutation(game_ids)
    test_games = set(shuffled[:n_test])
 
    # ~ is NOT operator in Polars expressions
    train_df = df.filter(~pl.col("game_id").is_in(test_games))
    test_df  = df.filter( pl.col("game_id").is_in(test_games))
 
    print(f"  Train: {len(train_df):,} moves from {len(game_ids) - n_test} games")
    print(f"  Test:  {len(test_df):,} moves from {n_test} games")
 
    return train_df, test_df


def train_model(
    train_df: pl.DataFrame, 
    test_df: pl.DataFrame,
) -> tuple[xgb.XGBRegressor, np.ndarray, np.ndarray]: 
    """Train XGBoost Regressor """
    
    X_train = train_df[FEATURE_COLS].to_numpy() 
    y_train = train_df[LABEL_COL].to_numpy() 
    X_test  = test_df[FEATURE_COLS].to_numpy()
    y_test  = test_df[LABEL_COL].to_numpy()
    
    print("\n Training XGBoost...")
    model = xgb.XGBRegressor( 
        n_estimators          = 500,
        max_depth             = 4,
        learning_rate         = 0.05,
        subsample             = 0.8,
        colsample_bytree      = 0.8,
        early_stopping_rounds = 20,
        eval_metric           = "mae",
        random_state          = 42,
        n_jobs                = -1,              
                             
    )
    
    model.fit(
        X_train, y_train,
        eval_set = [(X_test, y_test)],
        verbose  = 50,
    )

    print(f"  Stopped at tree {model.best_iteration} of 500")
 
    return model, X_test, y_test

def evaluate(model: xgb.XGBRegressor, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Evaluate model on test set.
 
    MAE: average error in centipawns — headline metric.
         If MAE=30, you're off by ~1/3 pawn on average.
 
    RMSE: penalizes large errors more than MAE.
          Useful for spotting catastrophic mispredictions.
 
    Label accuracy: even if centipawn number is slightly wrong,
                    do we predict the right category?
                    (good / inaccuracy / mistake / blunder)
                    This is what matters for the actual product UI.
    """
    preds = model.predict(X_test)
 
    mae  = float(np.mean(np.abs(preds - y_test)))
    rmse = float(np.sqrt(np.mean((preds - y_test) ** 2)))
 
    pred_labels   = np.array([to_label(p) for p in preds])
    actual_labels = np.array([to_label(a) for a in y_test])
    label_acc     = float(np.mean(pred_labels == actual_labels))
 
    # Per-class accuracy — useful for spotting if model ignores rare classes
    print(f"\n── Evaluation ──────────────────────────────────")
    print(f"  MAE:            {mae:.1f} centipawns")
    print(f"  RMSE:           {rmse:.1f} centipawns")
    print(f"  Label accuracy: {label_acc:.1%}")
    print(f"\n  Per-class accuracy:")
    for label_int, label_name in LABEL_NAMES.items():
        mask = actual_labels == label_int
        if mask.sum() > 0:
            class_acc = float(np.mean(pred_labels[mask] == actual_labels[mask]))
            count     = int(mask.sum())
            print(f"    {label_name:<12} {class_acc:.1%}  ({count} moves)")
    print(f"────────────────────────────────────────────────")
 
    return {"mae": mae, "rmse": rmse, "label_accuracy": label_acc}
 
 
# ── Feature importances ───────────────────────────────────────────────────────
 
def print_feature_importances(model: xgb.XGBRegressor) -> None:
    """
    Show which features the model relied on most.
 
    XGBoost tracks how often each feature was used to split a tree node
    and how much it reduced the loss. feature_importances_ sums to 1.0.
 
    High importance → this feature matters, engineer more like it.
    Low importance  → consider dropping it to simplify the model.
    This directly informs which features to focus on in the transformer.
    """
    importances = model.feature_importances_
    pairs = sorted(zip(FEATURE_COLS, importances), key=lambda x: -x[1])
 
    print("\n── Feature importances ─────────────────────────")
    for feat, imp in pairs:
        bar = "█" * int(imp * 40)
        print(f"  {feat:<25} {imp:.4f}  {bar}")
    print("────────────────────────────────────────────────")
 
 
# ── Save model ────────────────────────────────────────────────────────────────
 
def save_model(model: xgb.XGBRegressor, metrics: dict, models_dir: str) -> None:
    """
    Save model weights as JSON and metrics separately.
    JSON format is portable — can be loaded by any XGBoost version.
    """
    Path(models_dir).mkdir(parents=True, exist_ok=True)
 
    model_path   = f"{models_dir}/xgboost_centipawn.json"
    metrics_path = f"{models_dir}/xgboost_centipawn_metrics.json"
 
    model.save_model(model_path)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
 
    print(f"\nSaved model   → {model_path}")
    print(f"Saved metrics → {metrics_path}")
 
 
# ── Main ──────────────────────────────────────────────────────────────────────
 
def run(features_dir: str, labels_dir: str, models_dir: str) -> None:
    print("── Loading data ─────────────────────────────────")
    df = load_data(features_dir, labels_dir)
 
    print("\n── Splitting by game ────────────────────────────")
    train_df, test_df = split_by_game(df)
 
    model, X_test, y_test = train_model(train_df, test_df)
 
    metrics = evaluate(model, X_test, y_test)
    print_feature_importances(model)
    save_model(model, metrics, models_dir)
 
 
# ── CLI ───────────────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XGBoost training — Layer 3")
    parser.add_argument("--features-dir", type=str, default="data/features")
    parser.add_argument("--labels-dir",   type=str, default="data/labels")
    parser.add_argument("--models-dir",   type=str, default="models")
    args = parser.parse_args()
 
    run(args.features_dir, args.labels_dir, args.models_dir)
