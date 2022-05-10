from dataclasses import dataclass, field


@dataclass()
class TrainingParams:
    model_type: str = field(default="SVC")
    kernel_type: str = field(default="sigmoid")
    gamma: int = field(default=0.001)
    inverted_regularization: int = field(default=100)
    n_neighbors: int = field(default=20)