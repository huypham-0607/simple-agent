import torch

from torch import nn
from torch.utils.data import DataLoader
from utils import get_device
from models import LinearRegression
from trainer import Trainer
from gen_sine_data import SineData

device = get_device()

TIME_WINDOW = 20000
TRAIN_SPLIT = 0.8
TAU = 100

MAX_EPOCH = 100

def get_simple_autoregressor():
    train_data = SineData(True, TIME_WINDOW, TRAIN_SPLIT, TAU)
    test_data = SineData(False, TIME_WINDOW, TRAIN_SPLIT, TAU)

    lin_regressor = LinearRegression(100, 1)

    loss_fn = nn.MSELoss(reduction="mean")

    trainer = Trainer(50)

    trainer.fit(lin_regressor, train_data, loss_fn, 128, 0, 0.01)

    return lin_regressor

def main():
    model = get_simple_autoregressor()

if __name__ == "__main__":
    main()