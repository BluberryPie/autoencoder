from dataclasses import dataclass, field


@dataclass
class Config:
    random_seed: int = 42
    num_training_samples: int = 1_000
    noise_scale: float = 0.01
    hidden_dims: list[int] = field(default_factory=lambda: [32, 16])
    num_train_iterations: int = 10_000
    learning_rate: float = 1e-3
