##########################################################
# Evaluate a model with its training conditions 
##########################################################
import os
import torch
import numpy as np

from model import PhaseMaskModel
from utils import rs_tf_kernel, wavelengths_intensities
from config import PIXEL_SIZE
from training import all_accuracy

import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


def test_data(dataset, transforms_train):
    test_set = eval(f"datasets.{dataset}(root='../data', train=False, download=True, transform=transforms_train)")
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=10000, drop_last=True)
    return test_set, test_loader


if __name__ == '__main__':
    
    gpu_number = 0
    device = torch.device(f"cuda:{gpu_number}" if torch.cuda.is_available() else "cpu")
    print("current device:", device)

    paths = [''] # your path
    
    for path in paths:
        state = torch.load(path, map_location=device)
        
        transforms_train = transforms.Compose([transforms.ToTensor(),
                                               transforms.Pad(padding=(state['args'].pad,
                                                                       state['args'].pad, 
                                                                       state['args'].pad, 
                                                                       state['args'].pad))])

        test_set, test_loader = test_data(dataset=state['args'].dataset, transforms_train=transforms_train)

        # build the model and send it to the device
        SHAPE = state['args'].obj_shape + 2 * state['args'].pad 
           

        led = torch.ones((state['args'].led_size, state['args'].led_size), device=device) * state['args'].intens
        led_padding = (SHAPE - state['args'].led_size) // 2
        led = TF.pad(led, padding=(led_padding, led_padding, led_padding, led_padding))
        led = led.unsqueeze(0).unsqueeze(0)

        models = []
        for i in range(state['args'].pm_number):
            models.append(PhaseMaskModel(SHAPE, state['args'].base_wavelength, device).to(device))
        
        for i, model in enumerate(models):
            model.load_state_dict(state[f'models_{i}'])
            
        test_accuracy, confusion_matrix = all_accuracy(models,
                                                        state['args'],  
                                                        led,
                                                        state['args'].led_dist,
                                                        state['args'].num_wavelengths,
                                                        state['args'].sigma,
                                                        SHAPE,
                                                        test_loader,
                                                        device)

        print(f"Accuracy for {path} is:{test_accuracy:.4f}%") 
        path_split = path.split('/')
        path_to_save = path_split[0] + '/' + path_split[1]
        np.save(path_to_save + '/confusion_matrix.npy', confusion_matrix)

    