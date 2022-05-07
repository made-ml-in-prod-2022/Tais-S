from dataclasses import dataclass, field


@dataclass()
class TrainingParams:
    n_neighbors: int = field(default=5)