##########################################################
# Evaluate a model for different coherence conditions
##########################################################
import os
import torch
import torch.nn as nn
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
    
    num_wavelength_arr = [1, 3, 6, 12]
    sigma_arr = [0.009e15, 0.09e15, 0.2e15, 0.47e15]

    for path in paths:
        state = torch.load(path, map_location=device)
        transforms_train = transforms.Compose([transforms.Resize((state['args'].obj_shape, state['args'].obj_shape), 
                                                TF.InterpolationMode.NEAREST),
                                                transforms.ToTensor(),
                                                transforms.Pad(padding=(state['args'].pad,
                                                                        state['args'].pad, 
                                                                        state['args'].pad, 
                                                                        state['args'].pad))])
        
        test_set, test_loader = test_data(dataset=state['args'].dataset, transforms_train=transforms_train)

        # build the model and send it to the device
        SHAPE = state['args'].obj_shape + 2 * state['args'].pad 

        led = torch.ones((state['args'].led_size, state['args'].led_size), device=device) 
        led_padding = (SHAPE - state['args'].led_size) // 2
        led = TF.pad(led, padding=(led_padding, led_padding, led_padding, led_padding))
        led = led.unsqueeze(0).unsqueeze(0)

        models = []
        for i in range(state['args'].pm_number):
            models.append(PhaseMaskModel(SHAPE, state['args'].base_wavelength, device).to(device))
        
        for i, model in enumerate(models):
            model.load_state_dict(state[f'models_{i}'])
        
        for num_wavelengths, sigma in zip(num_wavelength_arr, sigma_arr):
            if num_wavelengths==1:
                led_dist_arr = [0.14e-2, 0.48e-2, 0.94e-2, 1.44e-2, 4.36e-2]
            else: 
                led_dist_arr = [0.15e-2, 0.5e-2, 1e-2, 1.55e-2, 5.5e-2]
            for led_dist in led_dist_arr:
                test_accuracy, _ = all_accuracy(models,
                                                state['args'],
                                                led,
                                                led_dist,
                                                num_wavelengths,
                                                sigma,
                                                SHAPE,
                                                test_loader,
                                                device)

                print(f"Accuracy for {path}, {led_dist} led dist and {num_wavelengths} wavelengths is:{test_accuracy:.4f}%") 
        