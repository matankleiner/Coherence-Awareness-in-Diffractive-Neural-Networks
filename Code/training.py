import time
import os
import torch
import numpy as np 
from tqdm import tqdm
from utils import rs_tf_kernel, patches_intensity, wavelengths_intensities
from layers import FreeSpaceProp, NonlinearLayer
from config import PIXEL_SIZE

def training(
        models,
        led,
        args,
        train_loader,
        test_loader,
        criterion,
        optimizer,
        scheduler,
        device,
):
    
    # training loop
    SHAPE = args.obj_shape + 2 * args.pad
    partial_led = torch.zeros(led.shape, device=device)
    ii, jj = torch.where(led[0, 0, :, :] > 0)

    for epoch in range(1, args.epochs + 1):
        for model in models:
            model.train()
        running_loss = 0.0
        optimizer.zero_grad()
        epoch_time = time.time()

        wavelengths_intensities_, wavelengths = wavelengths_intensities(args.base_wavelength,
                                                                    args.num_wavelengths,
                                                                    args.sigma,
                                                                    args.alpha)
        
        kernels = []
        led_kernels = []
        for wavelength in wavelengths:
            kernels.append(rs_tf_kernel(SHAPE, PIXEL_SIZE, wavelength, args.prop_dist, device))
            led_kernels.append(rs_tf_kernel(SHAPE, PIXEL_SIZE, wavelength, args.led_dist, device))
        # stack the filters, as each wavelength has a different filter
        KERNELS = torch.stack(kernels)
        LED_KERNELS = torch.stack(led_kernels)

        for a, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)  # send data to device
            labels = labels.to(device)

            intensity = torch.zeros((inputs.shape[0], SHAPE, SHAPE), device=device)
            partial_input = torch.zeros((inputs.shape[0], ii.shape[0], inputs.shape[2], inputs.shape[3]), dtype=torch.cfloat, device=device)
            
            for kernel, led_kernel, wavelength, wavelength_intensity in zip(KERNELS, LED_KERNELS, wavelengths, wavelengths_intensities_):
                free_space_led = FreeSpaceProp(led_kernel)
                free_space_prop = FreeSpaceProp(kernel)
                # creating a channel from each pixel of the light source
                for kk in range(ii.shape[0]):    
                    partial_led[0, 0, ii[kk], jj[kk]] = led[0, 0, ii[kk], jj[kk]]
                    led_f = free_space_led(partial_led)
                    led_f /= torch.max(torch.abs(led_f))
                    partial_input[:, kk, :, :] = (led_f * inputs * wavelength_intensity)[:, 0, :, :]
                    partial_led[0, 0, ii[kk], jj[kk]] = 0
            
                # forward pass
                x = free_space_prop(partial_input)
                for model in models:
                    x = model(x, wavelength)
                    x = free_space_prop(x)
                intensity = intensity + torch.sum((torch.abs(x))**2, dim=1)

            # calculate the loss
            patches_intensity_ = patches_intensity(intensity, device)
            loss = criterion(patches_intensity_, labels)

            # backward pass
            loss.backward()

            # gradient accumulation 
            if (a+1)%32: 
                optimizer.step()
                optimizer.zero_grad()
                
        running_loss += loss.data.item()
        # Normalizing the loss by the total number of train batches
        running_loss /= len(train_loader)
        # scheduler step 
        scheduler.step(running_loss)
        
        epoch_time = time.time() - epoch_time
        print(f"Epoch: {epoch:0>2}/{args.epochs} | Training Loss: {running_loss:.4f} | Epoch Time: {epoch_time:.2f} secs")

        if epoch % 5 == 0 or epoch == args.epochs:
            print('==> Saving model ...')
            state = {'epoch': epoch, 'args': args, 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}
            for i in range(len(models)):
                state[f'models_{i}'] = models[i].state_dict()
            os.makedirs(f'trials/{args.trial_name}/checkpoints', exist_ok=True)
            torch.save(state, f'./trials/{args.trial_name}/checkpoints/ckpt{epoch}.pth')

        if epoch % 10 == 0 or epoch == args.epochs:
            test_accuracy = accuracy(models, KERNELS, LED_KERNELS, wavelengths, wavelengths_intensities_, led, SHAPE, test_loader, device)
            print(f"Test Acc: {test_accuracy:.4f}%")


def accuracy(
    models,
    kernels,
    led_kernels,
    wavelengths,
    wavelengths_intensities_,
    led,
    shape,
    test_loader,
    device,
):

    partial_led = torch.zeros(led.shape, device=device)
    ii, jj = torch.where(led[0, 0, :, :] > 0)

    with torch.no_grad():
        for model in models:
            model.eval()
        total_images = 0
        total_correct = 0
        for data in test_loader:
            inputs, labels = data
            inputs = inputs.to(device)  # send data to device
            labels = labels.to(device)

            intensity = torch.zeros((inputs.shape[0], shape, shape), device=device)
            partial_input = torch.zeros((inputs.shape[0], ii.shape[0], inputs.shape[2], inputs.shape[3]), dtype=torch.cfloat, device=device)
            
            for kernel, led_kernel, wavelength, wavelength_intensity in zip(kernels, led_kernels, wavelengths, wavelengths_intensities_):
                free_space_led = FreeSpaceProp(led_kernel)
                free_space_prop = FreeSpaceProp(kernel)
                # creating a channel from each pixel of the light source
                for kk in range(ii.shape[0]):    
                    partial_led[0, 0, ii[kk], jj[kk]] = led[0, 0, ii[kk], jj[kk]]
                    led_f = free_space_led(partial_led)
                    led_f /= torch.max(torch.abs(led_f))
                    partial_input[:, kk, :, :] = (led_f * inputs * wavelength_intensity)[:, 0, :, :]
                    partial_led[0, 0, ii[kk], jj[kk]] = 0
            
                # forward pass
                x = free_space_prop(partial_input)
                for model in models:
                    x = model(x, wavelength)
                    x = free_space_prop(x)
                intensity = intensity + torch.sum((torch.abs(x))**2, dim=1)

            # calculate the loss
            patches_intensity_ = patches_intensity(intensity, device)
            _, predicted = torch.max(patches_intensity_.data, 1)
            total_images += labels.size(0)
            total_correct += (predicted == labels).sum().item()
        test_accuracy = total_correct / total_images * 100

    return test_accuracy


def all_accuracy(
    models,
    args,
    led,
    led_dist,
    num_wavelengths,
    sigma,
    shape,
    test_loader,
    device,
):
    
    partial_led = torch.zeros(led.shape, device=device)
    ii, jj = torch.where(led[0, 0, :, :] > 0)
   
    with torch.no_grad():
        for model in models:
            model.eval()
        total_images = 0
        total_correct = 0
        confusion_matrix = np.zeros([10, 10], int)
        data = next(iter(test_loader))
        all_inputs, all_labels = data
        all_inputs = all_inputs.to(device)  
        all_labels = all_labels.to(device)
        
        for i in tqdm(range(all_inputs.shape[0])):
            inputs = all_inputs[i].unsqueeze(0)
            labels = all_labels[i].unsqueeze(0)

            wavelengths_intensities_, wavelengths = wavelengths_intensities(args.base_wavelength,
                                                                        num_wavelengths,
                                                                        sigma,
                                                                        alpha=0.1)
            
            kernels = []
            led_kernels = []
            for wavelength in wavelengths:
                kernels.append(rs_tf_kernel(shape, PIXEL_SIZE, wavelength, args.prop_dist, device))
                led_kernels.append(rs_tf_kernel(shape, PIXEL_SIZE, wavelength, led_dist, device))
            # stack the filters, as each wavelength has a different filter
            KERNELS = torch.stack(kernels)
            LED_KERNELS = torch.stack(led_kernels)

            partial_input = torch.zeros((inputs.shape[0], ii.shape[0], inputs.shape[2], inputs.shape[3]), dtype=torch.cfloat, device=device)
         
            intensity = torch.zeros((inputs.shape[0], shape, shape), device=device)
            
            for kernel, led_kernel, wavelength, wavelength_intensity in zip(KERNELS, LED_KERNELS, wavelengths, wavelengths_intensities_):
                free_space_led = FreeSpaceProp(led_kernel)
                free_space_prop = FreeSpaceProp(kernel)
                # creating a channel from each pixel of the light source
                for kk in range(ii.shape[0]):    
                    partial_led[0, 0, ii[kk], jj[kk]] = led[0, 0, ii[kk], jj[kk]]
                    led_f = free_space_led(partial_led)
                    led_f /= torch.max(torch.abs(led_f))
                    partial_input[:, kk, :, :] = (led_f * inputs * wavelength_intensity)[:, 0, :, :]
                    partial_led[0, 0, ii[kk], jj[kk]] = 0
            
                # forward pass
                x = free_space_prop(partial_input)
                for model in models:
                    x = model(x, wavelength)
                    x = free_space_prop(x)
                intensity = intensity + torch.sum((torch.abs(x))**2, dim=1)

            patches_intensity_ = patches_intensity(intensity, device)
            _, predicted = torch.max(patches_intensity_.data, 1)
            total_images += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            for i, l in enumerate(labels):
                confusion_matrix[l.item(), predicted[i].item()] += 1
        test_accuracy = total_correct / total_images * 100

    return test_accuracy, confusion_matrix
