"""QRL trainers package — algorithm-specific training loops and factory dispatch."""

from qrl.trainers.dqn_trainer import train as train_dqn, DQNTrainer
from qrl.trainers.a2c_trainer import train_a2c, A2CTrainer
from qrl.trainers.pg_trainer import train_pg, PGTrainer
from qrl.trainers.factory import build_trainer

__all__ = [
    "train_dqn",
    "train_a2c",
    "train_pg",
    "A2CTrainer",
    "PGTrainer",
    "DQNTrainer",
    "build_trainer",
]
