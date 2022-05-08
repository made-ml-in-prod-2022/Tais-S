from dataclasses import dataclass, field


@dataclass()
class TrainingParams:
    kernel_type: str = field(default="sigmoid")
    gamma: int = field(default=0.001)
    inverted_regularization:int = field(default=100)