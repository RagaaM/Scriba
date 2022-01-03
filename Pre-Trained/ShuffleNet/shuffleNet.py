from PIL import Image
import numpy as np
from cnn_finetune import make_model
from torchvision import transforms
#import gradio as gr
import os
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import OrderedDict
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from customDataset import GlyphDataset
#%%
#Setting Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Hyperparameters
input_size= 10
in_channel = 1
num_classes = 10
learning_rate = 1e-3
batch_size = 2
num_epochs = 1
width=50
height =75
bias = True
stride= (1,1)
n_samples = 10
channels = 1
#input_dataset = range(10)
#%%

#Load Data
dataset = GlyphDataset(csv_file = 'GlyphDataset.csv', root_dir = 'GlyphDataset',
                        transform=transforms.ToTensor())
train_set, test_set = torch.utils.data.random_split(dataset, [7,3])
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)


#Model & fine tuning
model = torch.hub.load('pytorch/vision:v0.9.0', 'shufflenet_v2_x1_0', pretrained=True)


# def make_classifier(in_features, num_classes):
#     return nn.Sequential(
#         nn.Linear(input_size, 2),
#         nn.ReLU(),
#         nn.Linear(2,num_classes),
#         nn.Conv2d(1, 4,(3,3),(1,1))
#     )

# model = make_model('shufflenet_v2_x1_0', num_classes, pretrained=True, input_size=(50, 75), classifier_factory=make_classifier)
#%%
class Identity(nn.Module):
    def _init_(self):
        super(Identity, self)._init_()
        
    def forward(self, x):
        # x = self.fc1(x)
        # x = x.unsqueeze(dim=2)
        # x = F.relu(self.conv1d1(x))
        # x = x.squeeze()

        # x = self.fc2(x)
        return x
    
for param in model.parameters():
    param.requires_grad = False

model.avgpool=Identity()
model.classifier=nn.Sequential(nn.Linear(input_size, 2),
                                nn.ReLU(),
                                nn.Linear(2,num_classes),
                                nn.Conv2d(1, 4,(3,3),(1,1)))
model.to(device)

print(model)
model.eval()
#%%

#Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#%%
#Train Network

for epoch in range(num_epochs):
    losses = []
    
    for batch_idx, (data, targets) in enumerate(train_loader):
        #Get data to cuda if possible
        data=data.to(device=device)
        targets = targets.to(device=device)
       
        #Get to correct shape
        data = data.reshape(data.shape[0], -1)
        
        #Forward
        scores = model(data)
        loss = criterion(scores, targets)
        
        losses.append(loss.item())
        
        #Backward
        optimizer.zero_grad()
        loss.backward()
        
        #Gradient descent or Adam step
        optimizer.step()
        
    print(f'Cost at {epoch} is {sum(losses)/len(losses)}')
    
#%%    
#Check accuracy on training
def check_accuracy(loader, model):
    num_correct=0
    num_samples=0
    model.eval()
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0], -1)
            
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct+=(predictions==y).sum()
            num_samples+=predictions.size(0)
            
        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')
        
    model.train()
    
print('Checking accuracy on Training set')
check_accuracy(train_loader, model)

print('Checking accuracy on Test set')
check_accuracy(test_loader, model)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

