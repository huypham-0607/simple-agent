import torch

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0") # Prolly extremely device dependent lol
    else:
        return torch.device("cpu")
 