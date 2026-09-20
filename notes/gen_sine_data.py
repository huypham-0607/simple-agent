import torch
import torchvision

from utils import get_device
from torch.utils.data import Dataset

device = get_device()

time_window = 20000
time_step = 0.001
noise = 0.05

default_train_split = 0.8
default_tau = 100

class SineData(Dataset):
    @staticmethod
    def _gen_sine_data(T:int = time_window) -> tuple:
        time = torch.arange(1, T+1, dtype=torch.float32)
        print(len(time))
        x = torch.sin(time_step * time) + torch.randn(T) * noise
        return time, x
    
    def __init__(self, train:bool=True, T:int=time_window, train_split:float=default_train_split, tau:int=default_tau):
        super().__init__()
        self.T = T
        self.train_split = train_split
        self.tau=tau
        
        time, x = self._gen_sine_data(self.T)

        features = torch.stack([x[i : self.T-self.tau+i] for i in range(self.tau)],1)
        labels = x[self.tau:]
        time   = time[self.tau:]
        x      = x[self.tau:]

        print(len(features))
        print(len(features[0]))
        print(type(features[0]))
        print(features.shape)


        self.N = len(labels)
        num_train = int(train_split * self.N)
        if train:
            self.time = time[0:num_train].to(device)
            self.x = x[0:num_train].to(device)
            self.features = features[0:num_train].to(device)
            self.labels = labels[0:num_train].to(device)

        else:
            self.time = time[num_train:self.N].to(device)
            self.x = x[num_train:self.N].to(device)
            self.features = features[num_train:self.N].to(device)
            self.labels = labels[num_train:self.N].to(device)


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]