import argparse
import os
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from model import PhaseMaskModel
from utils import set_seed, rs_tf_kernel, wavelengths_intensities
from get_data import train_data, test_data
from training import training
from config import get_args, criterion, PIXEL_SIZE


if __name__ == '__main__':
    set_seed(seed=30)
    parsed = get_args()

    # load available device
    device = torch.device(f"cuda:{parsed.gpu_number}" if torch.cuda.is_available() else "cpu")
    print("current device:", device)

    transforms_train = transforms.Compose([transforms.ToTensor(),
                                           transforms.Pad(padding=(parsed.pad, parsed.pad, parsed.pad, parsed.pad))])

    train_set, train_loader = train_data(dataset=parsed.dataset, batch_size=parsed.batch_size, transforms_train=transforms_train)
    test_set, test_loader = test_data(dataset=parsed.dataset, batch_size=parsed.batch_size, transforms_train=transforms_train)

    # build the model and send it to the device
    SHAPE = parsed.obj_shape + 2 * parsed.pad 

    led = torch.ones((parsed.led_size, parsed.led_size), device=device) * parsed.intens
    led_padding = (SHAPE - parsed.led_size) // 2
    led = TF.pad(led, padding=(led_padding, led_padding, led_padding, led_padding))
    led = led.unsqueeze(0).unsqueeze(0)

    models = []
    for i in range(parsed.pm_number):
        models.append(PhaseMaskModel(SHAPE, parsed.base_wavelength, device).to(device))

    # optimizer
    optimizer = torch.optim.Adam(models[0].parameters(), lr=parsed.lr)
    for i in range(1, len(models)):
        optimizer.add_param_group({'params' : models[i].parameters()})
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)

    # continue training from a checkpoint
    if parsed.continue_training:
        checkpoint = torch.load(parsed.ckpt_path)
        for i, model in enumerate(models):
            model.load_state_dict(checkpoint[f'models_{i}'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        parsed.epoch = checkpoint['epoch']

    training(models, led, parsed, train_loader, test_loader, criterion, optimizer, scheduler, device)
    if not os.path.isdir(f'trials/{parsed.trial_name}'):
        os.mkdir(f'trials/{parsed.trial_name}')
