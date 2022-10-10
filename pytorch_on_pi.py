import time

import numpy as np
from torchvision import models, transforms

import cv2

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms

# Utility to visualize PyTorch network and shapes
from torchsummary import summary
from PIL import Image

# torch.backends.quantized.engine = 'qnnpack'
# ^ not supported on laptop????
# TODO : Need to train on certain frame width and height I think for params to line up
#frame_width = 280
#frame_height = 280
#cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

#cap = cv2.VideoCapture(1)
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 36)

"""while True:
    left = 700
    right = 1200
    top = 550
    bottom = 900
    ret, image = cap.read()
    # (height, width) = frame.shape[:2]
    #sky = frame[0:100, 0:200]
    print(image.shape)
    image = image[top:bottom, left:right]
    cv2.imshow('Video', image)
    print(image.shape)

    if cv2.waitKey(1) == 27:
        exit(0)
"""
classes = ('string1', 'string2', 'string3', 'string4', 'string5', 'string6', 'no_string')

transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([280,280]), # Resizing the image
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)) # 0.5 due to how torchvision data is in range [0,1]
    ])

##################################################################
"""
Getting our premade model
"""


# Note that PyTorch uses NCHW (samples, channels, height, width) image format convention
class ConvNet(nn.Module):
    def __init__(self, in_channels, num_filters, out_classes, kernel_size=3):
        super().__init__()
        # Conv2D layer with 'same' padding so image retains shape
        self.conv1 = nn.Conv2d(in_channels, num_filters, kernel_size, padding='same')
        self.drop = nn.Dropout(p=0.2)
        # self.dropout =
        # MaxPooling layer with 2x2 kernel size and stride 2
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(num_filters, num_filters * 2, kernel_size, padding='same')
        self.fc1 = nn.Linear(num_filters * 2 * 70 * 70, num_filters)
        self.fc2 = nn.Linear(num_filters, out_classes)

    def forward(self, x):
        x = x.to('cpu')
        # Non-linear ReLU activations between convolutional layers
        # Conv->ReLU->Pooling
        # 280x280 image -> 140x140 after pooling
        x = self.pool(self.drop(F.relu(self.conv1(x))))

        # 140x140 feature map -> 70x70 after pooling
        x = self.pool(self.drop(F.relu(self.conv2(x))))

        # Flatten all dimensions except batch (start_dim=1)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

# Todo: Magic number rn ... channels are rgb?
input_channels = 3
# Out classes = 7 ... 6 strings + No strings being pressed class
model = ConvNet(input_channels, num_filters=16, out_classes=7).to('cpu')
# Dummy inputs so we can plot a summary of the neural network's architecture and no. of parameters
#summary(model, input_size=(input_channels, 280, 280))


# LOAD MODEL FROM DISK
# Double check if model exists
#FOLDER_NAME = 'photos_colored_strings_cropped'
#FOLDER_NAME = 'photos_colored_strings_cropped_augmented'
#FOLDER_NAME = 'photos_colored_strings'
#FOLDER_NAME = 'photos_all_group_members'
FOLDER_NAME = 'photos_all_group_members_cropped_augmented'


model.load_state_dict(torch.load(os.getcwd() + '\\' + FOLDER_NAME + '.pth'))
print("Loaded model from disk!")
model.eval()
##################################################################

#net = models.quantization.mobilenet_v2(pretrained=True, quantize=True)
# jit model to take it from ~20fps to ~30fps
net = model
net = torch.jit.script(net)

started = time.time()
last_logged = time.time()
frame_count = 0

with torch.no_grad():
    while True:
        # read frame
        ret, image = cap.read()
        if not ret:
            raise RuntimeError("failed to read frame")

        # do something with output ...
        ####################################################################

        left = 700
        right = 1200
        top = 550
        bottom = 900
        image = image[top:bottom, left:right]
        cv2.imshow('Video', image)
        if cv2.waitKey(1) == 27:
            exit(0)
        # convert opencv output from BGR to RGB
        image = image[:, :, [2, 1, 0]]
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # preprocess
        input_tensor = transform(image)
        # TODO: NEED TO PREPEND EXTRA DIM None, CAUSE TRAINED ON  64 3 280 280 and images are just 3 280 280
        input_tensor = input_tensor[None, :]
        # run model
        output = net(input_tensor)
        ####################################################################
        # log model performance
        frame_count += 1
        now = time.time()
        if now - last_logged > 1:
            print(f"{frame_count / (now-last_logged)} fps")
            last_logged = now
            frame_count = 0
            top = list(enumerate(output[0].softmax(dim=0)))
            top.sort(key=lambda x: x[1], reverse=True)
            for idx, val in top[:7]:
                print(f"{val.item() * 100:.2f}% {classes[idx]}")
