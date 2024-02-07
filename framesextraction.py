#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import imutils
import numpy as np

import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

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

cap = cv2.VideoCapture(0)
top, right, bottom, left = 5, 990, 350, 650
#top, right, bottom, left = 10, 350, 225, 590
with mp_hands.Hands(
    min_detection_confidence=0.5,#detection sensitivity
    min_tracking_confidence=0.5) as hands:
    while True:
    # Read each frame from the webcam
    
        _, frame = cap.read()
        frame = imutils.resize(frame, width=1000)
        x , y, c = frame.shape

    ####################################################

        frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        # # to improve performance optionally mark the image as not writeable to pass by reference
        frame.flags.writeable = False
        results = hands.process(frame)

        # drawing hand annotations
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # if results.multi_hand_landmarks:
        #   for hand_landmarks in results.multi_hand_landmarks:
        #      mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    ###################################################
        # Flip the frame vertically
        #frame = cv2.flip(frame, 1)
        (height, width) = frame.shape[:2]
        roi = frame[top:bottom, left:right]
        # Show the final output
        #     cv2.imshow("Output", roi)
        image = cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
        #cv2.imshow("Output", image)
        #cv2.imshow("ROI", roi)
        #print("roi shape",roi.shape)


        # #skin Mask
        HSV_img = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_color = np.array([5, 50, 50], dtype="uint8")
        upper_color = np.array([15, 255, 255], dtype="uint8")
        #
        mask = cv2.inRange(HSV_img, lower_color, upper_color)
        # # # cv2.imshow('Mask 1', mask)
        # #
        # # # morphological operations,erosion followed by dilation
        kernel = np.ones((4, 4), np.uint8)
        erosion = cv2.erode(mask, kernel, iterations=1)
        dilation = cv2.dilate(erosion, kernel, iterations=1)
        # # # cv2.imshow('opening process', dilation)
        # #
        # # # gaussian blur
        blur = cv2.GaussianBlur(dilation, (5, 5), 0)
        #
        #
        # #blur1 = cv2.resize(blur, newsize)
        # # # cv2.imshow('Gausssian Blur', blur)
        # #
        #
        #
        # # # masking with original image
        mask2 = cv2.bitwise_and(roi, roi,mask=blur)
        cv2.imshow("Mask Image",mask2)

        #newsize = (224, 224)
        newsize = (128, 128)
        roi1 = cv2.resize(roi, newsize)
        color_convert = cv2.cvtColor(roi1,cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(color_convert)

        ################################################
        # with open('new_model/model.pickle', 'rb') as f:  # rb means read binary
        #     mp = pickle.load(f)

        train_path = 'D:/New/train'
        #train_path = 'C:/Users/Dev Prajapati/Desktop/CNN/final_marathi_mask_splitdataset/train'

        #train_path = 'C:/Users/Dev Prajapati/Desktop/CNN/final_marathi_non_mask_splitdataset/train'
        #train_path = 'C:/Users/Dev Prajapati/Desktop/CNN/nm_marathi_dataset/train'
        #train_path = 'C:/Users/Dev Prajapati/Desktop/CNN/sign_dataset/train_dataset'

        # categories
        root = pathlib.Path(train_path)
        classes = sorted([j.name.split('/')[-1] for j in root.iterdir()])

        ######
        # CNN Network

        class ConvNet(nn.Module):
            def __init__(self, num_classes=42):
                super(ConvNet, self).__init__()

                # Output size after convolution filter
                # ((w-f+2P)/s) +1

                # Input shape= (256,3,128,128)

                self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
                # Shape= (256,12,128,128)
                self.bn1 = nn.BatchNorm2d(num_features=12)
                # Shape= (256,12,128,128)
                self.relu1 = nn.ReLU()
                # Shape= (256,12,128,128)

                self.pool = nn.MaxPool2d(kernel_size=2)
                # Reduce the image size be factor 2
                # Shape= (256,12,64,64)

                self.conv2 = nn.Conv2d(in_channels=12, out_channels=20, kernel_size=3, stride=1, padding=1)
                # Shape= (256,20,64,64)
                self.relu2 = nn.ReLU()
                # Shape= (256,20,64,64)

                self.conv3 = nn.Conv2d(in_channels=20, out_channels=32, kernel_size=3, stride=1, padding=1)
                # Shape= (256,32,64,64)
                self.bn3 = nn.BatchNorm2d(num_features=32)
                # Shape= (256,32,64,64)
                self.relu3 = nn.ReLU()
                # Shape= (256,32,64,64)

                #self.fc = nn.Linear(in_features=64 * 64 * 32, out_features=num_classes)
                self.fc=nn.Linear(in_features=112 * 112 * 32,out_features=num_classes)

                # Feed forwad function

            def forward(self, input):
                output = self.conv1(input)
                output = self.bn1(output)
                output = self.relu1(output)

                output = self.pool(output)

                output = self.conv2(output)
                output = self.relu2(output)

                output = self.conv3(output)
                output = self.bn3(output)
                output = self.relu3(output)

                # Above output will be in matrix form, with shape (256,32,64,64)

                output = output.view(-1, 32 * 112 * 112)
                #output = output.view(-1, 32 * 64 * 64)

                output = self.fc(output)

                return output


        # In[5]:

        
        checkpoint = torch.load('D:/Sign-Language-Recognition--For-Marathi-Language-main/new_model/final_mask_marathi_best_checkpoint.model')

        #checkpoint = torch.load('old_model/ISL_best_checkpoint.model')
        #checkpoint = torch.load('new_model/final_non_mask_marathi_best_checkpoint.model')  # saved model
        #checkpoint = torch.load('new_model/nm_marathi_best_checkpoint.model')
        model = ConvNet(num_classes=42)
        model.load_state_dict(checkpoint)
        model.eval()
        ######

        # Transforms
        transformer = transforms.Compose([
        transforms.Resize((224, 224)),
        #transforms.Resize((128, 128)),
        transforms.ToTensor(),  # 0-255 to 0-1, numpy to tensors
        transforms.Normalize([0.5, .5, 0.5],  # 0-1 to [-1,1] , formula (x-mean)/std
                         [0.5, 0.5, 0.5])
        ])



        # prediction function
        def prediction(img, transformer):

        #image = Image.open(img_path)
        # print(image)

            image_tensor = transformer(img).float()

            image_tensor = image_tensor.unsqueeze_(0)

            if torch.cuda.is_available():
                image_tensor.cuda()

            input = Variable(image_tensor)

            output = model(input)  # will return array of probabilities

            index = output.data.numpy().argmax()

            pred = classes[index]

            return pred


        pred = prediction(pil_image,transformer)
        print(pred)
        image = cv2.putText(frame, pred, (right-200, bottom+60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2)#(0, 0, 255), 2) (255,255,255)
        cv2.imshow("output",image)
        ################################################

        if cv2.waitKey(1) == ord('q'):
            break
# release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()


# In[ ]:




