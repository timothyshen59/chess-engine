#Initializes WebDataset from TAR shards

from __future__ import annotations

import io
from pathlib import Path
from typing import Literal

import torch
import webdataset as wds
from torch.utils.data import DataLoader

def list_shards(root: str, split: str)-> list[str]: 
    root = Path(root) 
    shards = sorted(root.joinpath(split).glob(f"{split}-*.tar")) #Hardcoded 
    
    if not shards: 
        raise FileNotFoundError(f"No {split} shards found in {root /split }")

    return [str(shard) for shard in shards]

def load_pth(payload: bytes) -> dict[str, torch.Tensor]: 
    return torch.load(
        io.BytesIO(payload), 
        map_location="cpu", 
        weights_only=True, 
    )
    
def make_dataset(root: str | Path, split: str, *, shuffle_buffer: int = 2_048): 
    is_train = (split == "train") 
    
    dataset = wds.WebDataset(
        list_shards(root, split), 
        shardshuffle=is_train, 
        handler=wds.handlers.reraise_exception, 
    )
    
    if is_train: 
        dataset = dataset.shuffle(shuffle_buffer)
        
    return dataset.map_dict(pth=load_pth).map(lambda sample: sample["pth"])

def make_dataloader(
    root: str | Path, 
    split: str, 
    *, 
    batch_size: int, 
    num_workers: int = 0, 
    shuffle_buffer: int = 2_048, 
    pin_memory:bool = True, 
) -> DataLoader: 
    is_train = (split == "train")
    
    dataset = make_dataset(
        root = root, 
        split = split, 
        shuffle_buffer=shuffle_buffer,
    )
    
    loader_args = { 
        "batch_size": batch_size, 
        "num_workers": num_workers, 
        "pin_memory": pin_memory, 
        "drop_last": is_train,
    }
    
    if num_workers > 0: 
        loader_args["persistent_workers"] = True 
        loader_args["prefetch_factor"] = 2 
    
    return DataLoader(dataset, **loader_args)