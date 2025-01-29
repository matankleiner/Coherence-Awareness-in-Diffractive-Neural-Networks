import torch.nn as nn
from layers import PhaseMask_MultiModel


class PhaseMaskModel(nn.Module):
    def __init__(self, shape, base_wavelength, device):

        super(PhaseMaskModel, self).__init__()
        self.pm = PhaseMask_MultiModel(shape, base_wavelength, device)

    def forward(self, x, wavelength):
        return self.pm(x, wavelength)
