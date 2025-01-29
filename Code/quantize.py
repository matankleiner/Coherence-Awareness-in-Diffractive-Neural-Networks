##########################################################
# Evaluate a model with weights quantized to a different levels
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
    levels = []
    num_wavelengths_arr = []
    sigma_arr = []
    led_dist_1wavelengths_arr = []
    led_dist_more_arr = []
    
    for path in paths:
        for level in levels:
            for (num_wavelengths, sigma) in zip(num_wavelengths_arr, sigma_arr):
                    if num_wavelengths==1:
                        led_dist_arr = led_dist_1wavelengths_arr
                    else:
                        led_dist_arr = led_dist_more_arr
                    for led_dist in led_dist_arr:
                        state = torch.load(path, map_location=device)
                        
                        transforms_train = transforms.Compose([transforms.ToTensor(),
                                                               transforms.Pad(padding=(state['args'].pad,
                                                                                       state['args'].pad, 
                                                                                       state['args'].pad, 
                                                                                       state['args'].pad))])

                        test_set, test_loader = test_data(dataset=state['args'].dataset, transforms_train=transforms_train)

                        # build the model and send it to the device
                        SHAPE = state['args'].obj_shape + 2 * state['args'].pad 
                        SHAPE = state['args'].obj_shape + 2 * state['args'].pad 
                        wavelengths_intensities_, wavelengths = wavelengths_intensities(state['args'].base_wavelength,
                                                                                        state['args'].num_wavelengths,
                                                                                        state['args'].sigma,
                                                                                        state['args'].alpha)
                            
                        led = torch.ones((state['args'].led_size, state['args'].led_size), device=device)
                        led_padding = (SHAPE - state['args'].led_size) // 2
                        led = TF.pad(led, padding=(led_padding, led_padding, led_padding, led_padding))
                        led = led.unsqueeze(0).unsqueeze(0)
                        
                        models = []
                        for i in range(state['args'].pm_number):
                            models.append(PhaseMaskModel(SHAPE, state['args'].base_wavelength, device).to(device))
                        
                        for i, model in enumerate(models):
                            model.load_state_dict(state[f'models_{i}'])
                            ravel_pm = nn.Parameter(models[i].pm.weights % (2 * np.pi)).detach().cpu()
                            discrete_value = np.arange(0, 2*np.pi, np.pi/(level/2))
                            bins_pm = [np.digitize(ravel_pm[i,:], discrete_value) for i in range(ravel_pm.shape[0])]
                            discrete_ravel_pm = discrete_value[np.stack(bins_pm) - 1]
                            models[i].pm.weights = torch.nn.Parameter(torch.tensor(discrete_ravel_pm).to(device))
  
                        test_accuracy, _ = all_accuracy(models,
                                                        state['args'],
                                                        led,
                                                        led_dist,
                                                        num_wavelengths,
                                                        sigma,
                                                        SHAPE,
                                                        test_loader,
                                                        device)

                        print(f"Accuracy for {path}, sigma {sigma}, {num_wavelengths} wavelengths, {led_dist} led dist, and for {level} levels is:{test_accuracy:.4f}%") 
