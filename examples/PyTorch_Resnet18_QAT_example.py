#!/usr/bin/env python
# coding: utf-8

# In[27]:


from __future__ import print_function 
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy





# Top level data directory. Here we assume the format of the directory conforms 
#   to the ImageFolder structure
data_dir = "hymenoptera_data"

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]


# Number of classes in the dataset
num_classes = 2

# Batch size for training (change depending on how much memory you have)
batch_size = 8

# Number of epochs to train for 
num_epochs = 5

# Flag for feature extracting. When False, we finetune the whole model, 
#   when True we only update the reshaped layer params
feature_extract = True




def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    since = time.time()

    val_acc_history = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):


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

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history



def test_model(model, dataloaders, criterion, optimizer):
    best_acc = 0.0

    # Each epoch has a training and validation phase
    for phase in ['val']:
        model.eval()   # Set model to evaluate mode
        running_corrects = 0
        # Iterate over data.
        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            
            running_corrects += torch.sum(preds == labels.data)
        accuracy = running_corrects.double() / len(dataloaders[phase].dataset)
        return (format(accuracy, ".6f"))
#         print('Accuracy: {:4f}'.format(accuracy))
            
            




def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
            
            
            
def initialize_model(num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables
    model_ft = None
    input_size = 0

    model_ft = models.resnet18(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    input_size = 224
    
    return model_ft, input_size

# Initialize the model for this run
model_ft, input_size = initialize_model(num_classes, feature_extract, use_pretrained=True)

# Print the model we just instantiated
print ("Resnet18 model:\n")
print(model_ft)



# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

print("Initializing Datasets and Dataloaders...")

# Create training and validation datasets
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
# Create training and validation dataloaders
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



# Send the model to GPU
model_ft = model_ft.to(device)

# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are 
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)



# Setup the loss fxn

import torch
from torchvision import models
from aimet_torch.quantsim import QuantizationSimModel
criterion = nn.CrossEntropyLoss()



t1=time.perf_counter()

print ("\n")
print ("Normal fine-tuning of the model")
print ("\n")
# Normal model Training and evaluation

model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=2)

#set model to eval mode

model_ft.eval()

t2=time.perf_counter()


print("time taken for normal fine-tuning of the model", t2-t1)

print ("\n")
print ("Accuracy of fp32 model")
print (test_model(model_ft, dataloaders_dict, criterion, optimizer_ft))
print ("\n")

ab=""
def forward_pass(model, iterations):
    ab=test_model(model, dataloaders_dict, criterion, optimizer_ft)
print (ab)
    
#Initialize the Quantization Simulator
sim = QuantizationSimModel(model_ft, (1, 3, 224, 224))  

# print ("\n")
# print ("Simulated model:\n")
# print (sim.model)
# print ("\n")

#Compute the Quantization encodings
sim.compute_encodings(forward_pass, forward_pass_callback_args=400)

print ("\n")
print ("Accuracy of int8 model")
print (test_model(sim.model, dataloaders_dict, criterion, optimizer_ft))
print ("\n")

#set simulated model to train mode for Quatization aware training 

sim.model.train()

# Quantization aware training

t3=time.perf_counter()

model_ft_1, hist = train_model(sim.model, dataloaders_dict, criterion, optimizer_ft, num_epochs=5)

t4=time.perf_counter()

print ("\n")
print("time taken for QAT", t4-t3)
sim.model.eval()
print ("\n")
print ("Accuracy of quantized model after QAT")
print (test_model(sim.model, dataloaders_dict, criterion, optimizer_ft))

#save model
sim.export("./model/","quantized_model", input_shape=(1, 3, 224, 224))


# In[ ]:





# In[ ]:




