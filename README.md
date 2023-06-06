# Capstone Project

Capstone project at Northeastern University. 

Completed by: Eva Vrijen,  Katherine Poissant, Matas Suziedelis, Aaron Wachowiak, Shakti Katheria

Advisor: Bahram Shafai

We created a guitar attachment that allows the guitar to be strummed and plucked with only one hand. To handle the plucking, a model needed to be created that could determine of which of the strings was being held in order to pluck that string. To accomplish that, we created a dataset of images of the strings being pressed, image preprocessing/augmentation, and finally created a convolutional neural network. The CNN was uploaded on a Raspberry pi board and read in and analyzed live video input of the strings and sent the information to an Arduino board which controlled the hardware portion of plucking the string.

Dataset used - capstone_cuda/photos_perfectly_cropped

Model used code - capstone_cuda/transfer_learning-120-new-crop-keep-going.ipynb

Arduino code - capstone_cuda/MtlpServoRasAd/MtlpServoRasAd.ino

Pi to Arduino code - capstone_cuda/connect_software_to_pi.py
