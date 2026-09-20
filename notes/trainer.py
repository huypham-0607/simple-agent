import math
import torch
import matplotlib.pyplot as plt

from collections import OrderedDict
from torch import nn
from torch.utils.data import DataLoader
from utils import get_device

device = get_device()

class ProgressBoard:
    """Collect (x, y) points per label, averaging every_n raw points into one."""

    def __init__(self, xlabel = "epoch", xlim = None, figsize = (3.5, 2.5)):
        self.xlabel = xlabel
        self.xlim = xlim
        self.figsize = figsize
        self.raw_points = OrderedDict()
        self.data = OrderedDict()

    def draw(self, x, y, label, every_n = 1):
        """Buffer a point, emitting the mean of each every_n-sized bin."""
        if label not in self.raw_points:
            self.raw_points[label] = []
            self.data[label] = []

        points = self.raw_points[label]
        points.append((x, y))
        if len(points) != every_n:
            return

        mean = lambda vals: sum(vals) / len(vals)
        self.data[label].append(
            (mean([p[0] for p in points]), mean([p[1] for p in points]))
        )
        points.clear()

    def show(self):
        plt.figure(figsize = self.figsize)
        styles = ["-", "--", "-.", ":"]
        colors = ["C0", "C1", "C2", "C3"]
        for (label, line), ls, color in zip(self.data.items(), styles, colors):
            plt.plot(
                [p[0] for p in line],
                [p[1] for p in line],
                linestyle = ls,
                color = color,
                label = label
            )
        if self.xlim:
            plt.xlim(self.xlim)
        plt.xlabel(self.xlabel)
        plt.legend()
        plt.show()

class Trainer:
    def __init__(
        self,
        max_epoch = 5,
        grad_clip_val = 0,
        plot_train_per_epoch = 2,
        plot_valid_per_epoch = 1
    ):
        self.max_epoch = max_epoch
        self.grad_clip_val = grad_clip_val
        self.plot_train_per_epoch = plot_train_per_epoch
        self.plot_valid_per_epoch = plot_valid_per_epoch
        self.board = ProgressBoard()

    def clip_gradients(self, grad_clip_val, model):
        params = [p for p in model.parameters() if p.requires_grad]
        if not params:
            return
        norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
        if norm > grad_clip_val:
            for param in params:
                param.grad[:] *= grad_clip_val / norm

    def fit(
        self,
        model,
        data,
        loss_fn,
        val_data = None,
        batch_size = 128,
        num_workers = 0,
        shuffle = True,
        lr = 0.01,
    ):
        model.to(device)
        model.train()
        loader = DataLoader(
            dataset = data,
            batch_size = batch_size,
            shuffle = shuffle,
            num_workers = num_workers
        )
        val_loader = None
        if val_data is not None:
            val_loader = DataLoader(
                dataset = val_data,
                batch_size = batch_size,
                shuffle = False,
                num_workers = num_workers
            )
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=getattr(model, "lr", lr)
        )

        self.board = ProgressBoard(xlim = [0, self.max_epoch])
        train_batch_idx = 0
        num_train_batches = len(loader)
        every_n_train = int(num_train_batches / self.plot_train_per_epoch)
        every_n_val = int(
            len(val_loader) / self.plot_valid_per_epoch
        ) if val_loader is not None else 0

        for epoch in range(self.max_epoch):
            running_loss = 0
            data_count = 0

            loss_lst = []
            for bid, (X, y) in enumerate(loader):
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                
                pred = model(X)
                loss = loss_fn(pred, y)
                loss_lst.append(loss.item())

                loss.backward()
                if self.grad_clip_val:
                    self.clip_gradients(self.grad_clip_val, model)
                optimizer.step()

                self.board.draw(
                    train_batch_idx / num_train_batches,
                    math.exp(loss.item()),
                    "train_ppl",
                    every_n = every_n_train
                )
                train_batch_idx += 1

                running_loss += loss.item() * y.shape[0]
                data_count += y.shape[0]

            if val_loader is not None:
                model.eval()
                with torch.no_grad():
                    for X, y in val_loader:
                        X, y = X.to(device), y.to(device)
                        loss = loss_fn(model(X), y)
                        self.board.draw(
                            epoch + 1,
                            math.exp(loss.item()),
                            "val_ppl",
                            every_n = every_n_val
                        )
                model.train()

            epoch_loss = running_loss / data_count
            # print(running_loss, data_count)
            print(f"Loss for epoch {epoch}: {epoch_loss:.4f}")
