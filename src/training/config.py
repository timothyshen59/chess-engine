from dataclasses import dataclass 
from pathlib import Path 
import tomllib 

@dataclass 
class TrainConfig: 
    training_store: str 
    models_dir: str 
    
    epochs: int 
    batch_size: int 
    num_workers: int 
    shuffle_buffer: int 
    learning_rate: float 
    
    d_model: int 
    n_heads: int 
    n_layers: int 

def load_config(path: str | Path) -> TrainConfig: 
    with open(path, "rb") as f: 
        raw = tomllib.load(f) 
        
    paths = raw["paths"]
    training = raw["training"]
    model = raw["model"]
    
    return TrainConfig(
        training_store=paths["training_store"],
        models_dir=paths["models_dir"],
        epochs=training["epochs"],
        batch_size=training["batch_size"],
        num_workers=training["num_workers"],
        shuffle_buffer=training["shuffle_buffer"],
        learning_rate=training["learning_rate"],
        d_model=model["d_model"],
        n_heads=model["n_heads"],
        n_layers=model["n_layers"],
    )