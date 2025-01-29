import argparse
import torch.nn as nn

SAMPLES = 5000 
PIXEL_SIZE = 10e-6 

# loss criterion
criterion = nn.CrossEntropyLoss()

def get_args():
    parser = argparse.ArgumentParser()
    
    # optimization hyperparameters 
    parser.add_argument('--batch_size', help='batch size', default=1, type=int)
    parser.add_argument('--lr', help='learning rate', default=1e-2, type=float)
    parser.add_argument('--epochs', help='number of epochs', default=50, type=int)
    
    # optical properties 
    parser.add_argument('--base_wavelength', help='', default=550e-9, type=float)
    parser.add_argument('--num_wavelengths', help='the number of wavelengths in the wavelengths sequence', default=6, type=int)
    parser.add_argument('--prop_dist', help='propagation distance between layers, in meter', default=5e-2, type=float)
    parser.add_argument('--led_dist', help='propagation distance between the led and the object, in meter', default=1e-2, type=float)
    parser.add_argument('--led_size', help='led size, in pixels', default=12, type=int)
    parser.add_argument('--alpha', help='', default=0.1, type=float)
    parser.add_argument('--sigma', help='', default=0.9e14, type=float)
    parser.add_argument('--intens', help='', default=1, type=int)
    
    # trial properties
    parser.add_argument('--obj_shape', help='object shape', default=28, type=int)
    parser.add_argument('--pad', help='padding', default=136, type=int)
    parser.add_argument('--pm_number', help='phase mask number', default=2, type=int)
    parser.add_argument('--window', help='partial incoherent window', default=1, type=int)
    parser.add_argument('--gaussian_std', help='gassuian std for partial incohrent blur sources propagation', default=1, type=float)

    # nonlinear modeling 
    parser.add_argument('--nonlinear', help='adding a nonlinear layer as part of the all optical neural network', action='store_true')

    # dataset to use 
    parser.add_argument('--dataset', help='dataset to use', default='MNIST', type=str)
    
    # trial configurations      
    parser.add_argument('--trial_name', help='the trial name', default='incoherent_trial', type=str)
    parser.add_argument('--gpu_number', help='gpu to use', default=0, type=int) 
    parser.add_argument('--raffles', default=300, type=int)

    # continue training 
    parser.add_argument('--continue_training', help='continue training from a checkpoint', action='store_true')
    parser.add_argument('--ckpt_path', help='path to checkpoint', type=str)
    
    return parser.parse_args()
