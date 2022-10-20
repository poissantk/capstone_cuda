import time

import numpy as np
from torchvision import models, transforms

import cv2

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms
from torchvision.models.vgg import VGG16_Weights


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
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 36)
crop = True
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
classes = ('no_string','string1', 'string2', 'string3', 'string4', 'string5', 'string6')

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


from torchvision import models

model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).to("cpu")

# Freeze early layers
for param in model.parameters():
    param.requires_grad = False

n_inputs = model.classifier[6].in_features

# Add on classifier
n_classes = 7
model.classifier[6] = nn.Sequential(
    nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.4),
    nn.Linear(256, n_classes), nn.LogSoftmax(dim=1))

def get_pretrained_model(model_name):
    """Retrieve a pre-trained model from torchvision

    Params
    -------
        model_name (str): name of the model (currently only accepts vgg16 and resnet50)

    Return
    --------
        model (PyTorch model): cnn

    """

    if model_name == 'vgg16':
        model = models.vgg16(pretrained=True)

        # Freeze early layers
        for param in model.parameters():
            param.requires_grad = False
        n_inputs = model.classifier[6].in_features

        # Add on classifier
        model.classifier[6] = nn.Sequential(
            nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, n_classes), nn.LogSoftmax(dim=1))

    return model

model = get_pretrained_model('vgg16')



# LOAD MODEL FROM DISK
# Double check if model exists
#FOLDER_NAME = 'photos_colored_strings_cropped'
#FOLDER_NAME = 'photos_colored_strings_cropped_augmented'
# FOLDER_NAME = 'photos_colored_strings'
#FOLDER_NAME = 'photos_all_group_members'
#FOLDER_NAME = 'photos_all_group_members_cropped_augmented'
FOLDER_NAME = "vgg16-transfer-4.pt"


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
        if crop:
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
