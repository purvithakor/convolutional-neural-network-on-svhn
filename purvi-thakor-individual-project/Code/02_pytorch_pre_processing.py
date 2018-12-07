# This file needs to be run in your working directory. From there, this code will create a folder called svhn_pytorch
# where all the data will be loaded and converted to images.
# The folder structure needed for Dataloader will also be created using this file.


import warnings
warnings.filterwarnings("ignore")

import os
import wget
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import matplotlib
import matplotlib.pyplot as plt
import time
from PIL import Image

print("\nThe current working directory is: " +str(os.getcwd())) # Current working directory
########################################################################################################################
# Create a folder (svhn_pytorch) in your current working directory if not present

folder_name = 'svhn_pytorch'

print("")
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
    os.chmod(folder_name, 0o777)
    print('Directory ' +str(folder_name)+' created!')
else:
    print('Directory ' +str(folder_name)+ ' already exists!')

########################################################################################################################
# Create two folders train and test in svhn_pytorch in your current working directory if not present

folder_names = ['svhn_pytorch/train', 'svhn_pytorch/test']

print("")
for folder_name in folder_names:
    filepath = os.getcwd() + "/" + folder_name

    if not os.path.exists(filepath):
        os.makedirs(filepath)
        os.chmod(folder_name, 0o777)
        print('Directory ' +str(folder_name)+' created!')
    else:
        print('Directory ' +str(folder_name)+ ' already exists!')

########################################################################################################################
# Creating the label folder structure

label_list = ['label_0','label_1','label_2','label_3','label_4','label_5','label_6','label_7','label_8','label_9']

#Creating Label folders in svhn_pytorch/train

os.chdir('./svhn_pytorch/train/') # go to ./svhn_pytorch/train to create label folders

print("")
for folder_name in label_list:
    filepath = os.getcwd() + "/" + folder_name

    if not os.path.exists(filepath):
        os.makedirs(filepath)
        os.chmod(folder_name, 0o777)
        print('Directory ' +str(folder_name)+' created!')
    else:
        print('Directory ' +str(folder_name)+ ' already exists!')
print("\nFolder structure for train images done.")

os.chdir("..")  #goes back to svhn_pytorch folder

#Creating Label folders in svhn_pytorch/test

os.chdir('./test/') # go to ./test to create the label folders

print("")
for folder_name in label_list:
    filepath = os.getcwd() + "/" + folder_name
    if not os.path.exists(filepath):
        os.makedirs(filepath)
        os.chmod(folder_name, 0o777)
        print('Directory ' +str(folder_name)+' created!')
    else:
        print('Directory ' +str(folder_name)+ ' already exists!')
print("\nFolder structure for test images done.")

########################################################################################################################
os.chdir("..")  #go back to svhn_pytorch

# DOWNLOADING PROCESSED .mat FILES FROM BUCKET on your instance in the svhn_pytorch folder
# This puts test_processed.mat & train_processed.mat files in your svhn_pytorch folder from the bucket

filename_list = ['test_processed.mat', 'train_processed.mat'] # filenames of .mat files on the bucket
# print("")
for filename in filename_list:
    filepath = './' + filename

    if not os.path.exists(filepath):
        print('\nDownloading ' +str(filename)+  ' file')
        url = 'https://storage.googleapis.com/1_deep_learning_final_project_group_1/processed_files/' + filename
        wget.download(url, filepath)
    else:
        print('\nFile ' + str(filename) + ' already exists.')

########################################################################################################################

#Load TRAIN DATA from instance to svhn_pytorch folder on your cloud

train_processed = loadmat('./train_processed.mat')
# train_processed is a dictionary of length 5, we are only interested in x_train and y_train

# extract x_train from dictionary
x_train_mat = train_processed['x_train']  #shape of x_train_mat = (32,32,3,504336)

# extract y_train from dictionary
y_train_mat = train_processed['y_train']  #shape of y_train_mat = (504336,1)

print("")
print("\nProcessed train files loaded!")
del(train_processed)
########################################################################################################################

#Load TEST DATA from instance to current environment

test_processed = loadmat('./test_processed.mat')
# test_processed is a dictionary of length 5, we are only interested in x_test and y_test

# extract x_test from dictionary
x_test_mat = test_processed['x_test']  #shape of x_test_mat = (32,32,3,126084)

# extract y_test from dictionary
y_test_mat = test_processed['y_test']  #shape of y_test_mat = (126084,1)

print("Processed test files loaded!")
del(test_processed)
#######################################################################################################################

#We need to convert the .mat files to images  and segregate the images in their appropriate label folder (to be used
#in DataLoader)

#CONVERTING TEST MAT TO IMAGES
txt_test_file = open("./test.txt", "w") # creating a text file just for reference

tot_test_img = len(x_test_mat[1,1,1,:])

for i in range(tot_test_img) :
    label = y_test_mat[i].item()
    Image.fromarray(x_test_mat[:,:,:,i]).save('./test/label_'+str(label)+'/image'+str(i)+'.jpg')
    txt_test_file.write("test/label_"+str(label)+"/image"+str(i)+".jpg="+str(label)+"\n")

txt_test_file.close()

print("\nTest images segregated!")

#CONVERTING TRAIN MAT TO IMAGES
txt_train_file = open("./train.txt", "w")

tot_train_img = len(x_train_mat[1,1,1,:])

for i in range(tot_train_img) :
    label = y_train_mat[i].item()
    Image.fromarray(x_train_mat[:,:,:,i]).save('./train/label_'+str(label)+'/image'+str(i)+'.jpg')
    txt_train_file.write("train/label_"+str(label)+"/image"+str(i)+".jpg="+str(label)+"\n")

txt_train_file.close()

print("Train images segregated!")

os.chdir("..") # go back to the folder where the codes are
print("\nThe current working directory is: " +str(os.getcwd()))

#########################################################  END  ########################################################
