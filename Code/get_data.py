import torch
import torchvision.datasets as datasets
from config import SAMPLES

  
def train_data(dataset, batch_size, transforms_train):
    train_set = eval(f"datasets.{dataset}(root='../data', train=True, download=True, transform=transforms_train)")
    samples = torch.randperm(train_set.data.shape[0])[:SAMPLES]
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, drop_last=True, sampler=samples)
    return train_set, train_loader

def test_data(dataset, batch_size, transforms_train):
    test_set = eval(f"datasets.{dataset}(root='../data', train=False, download=True, transform=transforms_train)")
    samples = torch.randperm(test_set.data.shape[0])[:SAMPLES//10]
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, drop_last=True, sampler=samples)
    return test_set, test_loader
