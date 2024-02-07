#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
from torchvision.transforms import transforms
import numpy as np
from torch.autograd import Variable
from torchvision.models import squeezenet1_1
import torch.functional as F
from io import open
import os
from PIL import Image
import pathlib
import glob
import cv2
import pickle


# In[2]:


train_path='C:/Users/Dev Prajapati/Desktop/CNN/marathi_dataset/train_dataset'
pred_path='C:/Users/Dev Prajapati/Desktop/CNN/marathi_dataset/pred_dataset'


# In[3]:


#categories
root=pathlib.Path(train_path)
classes=sorted([j.name.split('/')[-1] for j in root.iterdir()])


# In[4]:


#CNN Network


class ConvNet(nn.Module):
    def __init__(self,num_classes=42):
        super(ConvNet,self).__init__()
        
        #Output size after convolution filter
        #((w-f+2P)/s) +1
        
        #Input shape= (256,3,128,128)
        
        self.conv1=nn.Conv2d(in_channels=3,out_channels=12,kernel_size=3,stride=1,padding=1)
        #Shape= (256,12,128,128)
        self.bn1=nn.BatchNorm2d(num_features=12)
        #Shape= (256,12,128,128)
        self.relu1=nn.ReLU()
        #Shape= (256,12,128,128)
        
        self.pool=nn.MaxPool2d(kernel_size=2)
        #Reduce the image size be factor 2
        #Shape= (256,12,64,64)
        
        
        self.conv2=nn.Conv2d(in_channels=12,out_channels=20,kernel_size=3,stride=1,padding=1)
        #Shape= (256,20,64,64)
        self.relu2=nn.ReLU()
        #Shape= (256,20,64,64)
        
        
        
        self.conv3=nn.Conv2d(in_channels=20,out_channels=32,kernel_size=3,stride=1,padding=1)
        #Shape= (256,32,64,64)
        self.bn3=nn.BatchNorm2d(num_features=32)
        #Shape= (256,32,64,64)
        self.relu3=nn.ReLU()
        #Shape= (256,32,64,64)
        
        
        self.fc=nn.Linear(in_features=112*112*32,out_features=num_classes)
        #self.sf=nn.Softmax(in_features=112 * 112 * 32,out_features=num_classes)
        
        
        
        #Feed forwad function
        
    def forward(self,input):
        output=self.conv1(input)
        output=self.bn1(output)
        output=self.relu1(output)
            
        output=self.pool(output)
            
        output=self.conv2(output)
        output=self.relu2(output)
            
        output=self.conv3(output)
        output=self.bn3(output)
        output=self.relu3(output)
            
            
            #Above output will be in matrix form, with shape (256,32,64,64)
            
        output=output.view(-1,32*112*112)
            
            
        output=self.fc(output)
            
        return output
            
        


# In[5]:


checkpoint=torch.load('new_model/marathi_best_checkpoint.model')#saved model
model=ConvNet(num_classes=42)
model.load_state_dict(checkpoint)
model.eval()


# In[6]:


# #saving in pickel file
#import pickle
# with open('model.pickle','wb') as f:#wb means write binary
#     pickle.dump(model,f)


# In[7]:


with open('new_model/model.pickle','rb') as f:#rb means read binary
    mp = pickle.load(f)


# In[8]:


#Transforms
transformer=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),  #0-255 to 0-1, numpy to tensors
    transforms.Normalize([0.5,.5,0.5], # 0-1 to [-1,1] , formula (x-mean)/std
                        [0.5,0.5,0.5])
])


# In[18]:


#prediction function
def prediction(img_path,transformer):
    
    image=Image.open(img_path)
    #print(image)
    
    
    image_tensor=transformer(image).float()
    
    
    image_tensor=image_tensor.unsqueeze_(0)
    
    if torch.cuda.is_available():
        image_tensor.cuda()
        
    input=Variable(image_tensor)
    
    
    output=mp(input)#will return array of probabilities
    
    index=output.data.numpy().argmax()
    
    pred=classes[index]
    
    return pred
    


# In[22]:


images_path=glob.glob(pred_path+'/*.jpg')


# In[ ]:





# In[23]:


pred_dict={}

for i in images_path:
    pred_dict[i[i.rfind('/')+1:]]=prediction(i,transformer)


# In[24]:


print(pred_dict)


# In[ ]:




