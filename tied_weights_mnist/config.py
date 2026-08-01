from dataclasses import dataclass


@dataclass
class Config:
    data_dir: str = "data"
    batch_size: int = 32
