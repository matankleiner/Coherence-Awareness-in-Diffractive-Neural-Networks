import torch
import torch.nn as nn
from numpy import pi


class FreeSpaceProp(nn.Module):
    # The angular spectrum method with Rayleigh–Sommerfeld diffraction formulation
    def __init__(self, kernel_fft):
        super().__init__()
        self.kernel_fft = kernel_fft

    def forward(self, x):
        x_fft = torch.fft.fft2(torch.fft.fftshift(x, dim=(2,3)))
        out = torch.fft.ifftshift(torch.fft.ifft2(x_fft * self.kernel_fft), dim=(2,3))
        return out


class PhaseMask_MultiModel(nn.Module):

    def __init__(self, shape, base_wavelength, device):

        super().__init__()
        self.weights = torch.nn.Parameter(torch.randn(shape, shape))
        self.base_wavelength = base_wavelength
        self.device = device

    def forward(self, x, wavelength):
        # the wave propagation through the phase mask is different for each wavelength, therefor
        # a different phase mask needed for each wavelength
        pm = torch.exp(1j * self.weights * (self.base_wavelength / wavelength)).unsqueeze(0).unsqueeze(0).to(self.device)
        return x * pm
    

class NonlinearLayer(nn.Module):
    # taken from "Fourier-space Diffractive Deep Neural Network", T. Yan et-al, PRL, 2019
    # https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.123.023901
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, x, intensity):
        return x * (torch.exp(1j * pi * (intensity / (1 + intensity)))).to(self.device)
    