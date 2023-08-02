import os
import time, copy
import numpy as np 

import torch
import torch.nn as nn
from tqdm import tqdm


# Training loop for spot classification
def train_spotwise(model, dataloaders, criterion, optimizer, num_epochs, 
    outfile=None, display=False):
    since = time.time()

    val_history, train_history = [], []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # GPU support
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1), flush=True)
        print('-' * 10, flush=True)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Optionally display epoch progress to stdout
            if display:
                iterator = tqdm(dataloaders[phase])
            else:
                iterator = dataloaders[phase]

            # Iterate over data.
            for inputs, labels in iterator:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc), flush=True)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

                if outfile is not None:
                	torch.save(model.state_dict(), outfile)
            if phase == 'val':
                val_history.append(epoch_loss)
            else:
                train_history.append(epoch_loss)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60), flush=True)
    print('Best val Acc: {:4f}'.format(best_acc), flush=True)

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_history, train_history
    
# Training loop for grid registration
def train_gridwise(model, dataloaders, criterion, optimizer, num_epochs=25, outfile=None, 
    f_opt=None, accum_iters=1):
    since = time.time()

    train_history, val_history = [], []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # GPU support
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1), flush=True)
        print('-' * 10, flush=True)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            # Turn off batch normalization/dropout for patch classifier
            model.patch_classifier.eval()
            
            running_loss = 0.0
            running_corrects = 0
            running_foreground = 0

            # Iterate over data.
            for batch_ind, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs, then filter for foreground patches (label>0).
                    # Use only foreground patches in loss/accuracy calulcations.
                    outputs = model(inputs)

                    # Outputs: (batch, classes, d1, d2)
                    # Labels: (batch, d1, d2)
                    assert outputs.shape[2]==labels.shape[1] and outputs.shape[3]==labels.shape[2], "Output tensor does not match label dimensions!"

                    outputs = outputs.permute((0,2,3,1))
                    outputs = torch.reshape(outputs, (-1, outputs.shape[-1]))
                    labels = torch.reshape(labels, (-1,))
                    outputs = outputs[labels > 0]
                    labels = labels[labels > 0]
                    labels -= 1 # Foreground classes range between [1, N_CLASS].

                    loss = criterion(outputs, labels) / accum_iters
                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        
                        if batch_ind % accum_iters == 0:
                            optimizer.step()
                            optimizer.zero_grad()
                            if f_opt is not None:
                                f_opt.step()
                                f_opt.zero_grad()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                running_foreground += len(labels)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / running_foreground

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc), flush=True)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                if outfile is not None:
                    torch.save(model.state_dict(), outfile)
                    if f_opt is not None:
                        torch.save({
                            'g_opt':optimizer.state_dict(),
                            'f_opt':f_opt.state_dict()
                        }, os.path.splitext(outfile)[0]+".opt")
                    else:
                        torch.save(optimizer.state_dict(), os.path.splitext(outfile)[0]+".opt")
            if phase == 'val':
                val_history.append(epoch_loss)
            else:
                train_history.append(epoch_loss)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60), flush=True)
    print('Best val Acc: {:4f}'.format(best_acc), flush=True)

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_history, train_history
