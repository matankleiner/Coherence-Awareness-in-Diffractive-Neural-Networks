import torch
import numpy as np
import random


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rs_tf_kernel(shape, pixel_size, wavelength, dist, device):
    size = shape * pixel_size
    fx = torch.arange((-1 / (2 * (pixel_size))), (1 / (2 * (pixel_size))), (1/(size)))
    Fx, Fy = torch.meshgrid(fx, fx, indexing="ij")
    k = 2 * np.pi / wavelength
    H = torch.exp(1j * k * dist * torch.sqrt(1 - (wavelength * Fx)**2 - (wavelength * Fy)**2))
    H = H * torch.where(Fx**2 + Fy**2 < (1/wavelength)**2, 1, 0)
    return torch.fft.fftshift(H).to(device)


def wavelengths_intensities(base_wavelength, num_wavelengths, sigma, alpha):
    c = 3e8 
    spectrum_step = 10e-11
    spectrum_s = 300e-9
    spectrum_e = 800e-9
    lambda_arr = np.arange(spectrum_s, spectrum_e, spectrum_step)
    
    const_1 = ((2 * np.pi * c) / (base_wavelength))**2
    const_2 = 2 * sigma ** 2
    function_in_lambda = const_1 * (((base_wavelength - lambda_arr) / lambda_arr) ** 2) / const_2
    exp_lambda = np.exp(-1 * function_in_lambda)
    max_exp_lambda = exp_lambda[np.argmax(exp_lambda)]

    wavelengths_intensity = []
    wavelengths_val = []
    while len(wavelengths_val) < num_wavelengths:
        index = np.random.randint(0, len(exp_lambda))
        if exp_lambda[index] > alpha * max_exp_lambda:
            wavelengths_intensity.append(exp_lambda[index])
            wavelengths_val.append(lambda_arr[index])
    wavelengths_intensity = torch.tensor(wavelengths_intensity)
    normalize_intensity = wavelengths_intensity / torch.sum(wavelengths_intensity)

    return torch.sqrt(normalize_intensity), wavelengths_val


def patches_intensity(intensity, device):
    B = intensity.shape[0]
    patches_intensity = torch.zeros((B, 10)).to(device)

    if intensity.shape[-1] == 300:
        step = 60 
        start1 = 30 
        dist1 = 80
        start2 = 5 
        dist2 = 70
    else: 
        raise Exception("Not a valid shape")

    patches_intensity[:, 0] = torch.sum(intensity[:, start1:start1+step, start1:start1+step], dim=(1,2))
    patches_intensity[:, 1] = torch.sum(intensity[:, start1:start1+step, start1+dist1:start1+dist1+step], dim=(1,2))
    patches_intensity[:, 2] = torch.sum(intensity[:, start1:start1+step, start1+2*dist1:start1+2*dist1+step], dim=(1,2))
    patches_intensity[:, 3] = torch.sum(intensity[:, start1+dist1:start1+dist1+step, start2:start2+step], dim=(1,2))
    patches_intensity[:, 4] = torch.sum(intensity[:, start1+dist1:start1+dist1+step, start2+dist2:start2+dist2+step], dim=(1,2))
    patches_intensity[:, 5] = torch.sum(intensity[:, start1+dist1:start1+dist1+step, start2+2*dist2:start2+2*dist2+step], dim=(1,2))
    patches_intensity[:, 6] = torch.sum(intensity[:, start1+dist1:start1+dist1+step, start2+3*dist2:start2+3*dist2+step], dim=(1,2))
    patches_intensity[:, 7] = torch.sum(intensity[:, start1+2*dist1:start1+2*dist1+step, start1:start1+step], dim=(1,2))
    patches_intensity[:, 8] = torch.sum(intensity[:, start1+2*dist1:start1+2*dist1+step, start1+dist1:start1+dist1+step], dim=(1,2))
    patches_intensity[:, 9] = torch.sum(intensity[:, start1+2*dist1:start1+2*dist1+step, start1+2*dist1:start1+2*dist1+step], dim=(1,2))

    return patches_intensity
