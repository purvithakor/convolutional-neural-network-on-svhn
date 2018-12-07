# Best model for CNN

import matplotlib
##matplotlib.use('Qt4Agg')
##matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import os
import wget
import numpy as np
from scipy.io import loadmat
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import time
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report,confusion_matrix

########################################################################################################################
start = time.time()

#Setting the Hyper Parameters

num_epochs = 10
batch_size = 500
learning_rate = 0.001

########################################################################################################################
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
dtype = torch.float

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

print("The device being used is: " +str(device))

########################################################################################################################
print("\nThe current working directory is: " +str(os.getcwd()))

os.chdir("./svhn_pytorch")  #changing the directory from current to svhn_pytorch

print("\nChanged the working directory to: "+str(os.getcwd()))

########################################################################################################################

print("\nThe processed images will be converted into DataLoader for Train & Test")

data_transform = transforms.Compose([transforms.ToTensor()])  #[0,255] => [0,1]

#train dataloader
train_dataset = torchvision.datasets.ImageFolder('./train', transform=data_transform )
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, )

#test dataloader
test_dataset = torchvision.datasets.ImageFolder('./test', transform=data_transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("\nData is ready!")

########################################################################################################################
#CNN architecture

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding =0))
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding =0))
        nn.Dropout(p=0.2),
        self.fc1 = nn.Linear(in_features=6*6*32, out_features=10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out

########################################################################################################################

cnn = CNN().to(device)
cnn = cnn.cuda()

########################################################################################################################

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

# Training the Model

cnn.train()  #to include dropouts in training
loss_list = []

for epoch in range(num_epochs):
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())
        # reset the gradient
        optimizer.zero_grad()
        # forward pass
        outputs = cnn(images)
        # loss for this batch
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())
        # backward
        loss.backward()
        # Update parameters based on backpropogation
        optimizer.step()
        if (batch_idx + 1) % 100 == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                  % (epoch + 1, num_epochs, batch_idx + 1, len(train_dataset) // batch_size, loss.item()))

plt.figure(1)
plt.loglog(np.array(loss_list))
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend("Log Loss across iterations")
##plt.show(block=False)
plt.show()

########################################################################################################################
#Testing the Model

correct = 0
total = 0
correct_lst =[]

preds = []  # empty array to store the predicted values of 126,084 images
labs = []  # empty array to store the labels of 126,084 images

cnn.eval() #to exclude dropouts in testing

for images, labels in test_loader:
   images = images.to(device, dtype=torch.float)
   outputs = cnn(images)
   _, predicted = torch.max(outputs.data, 1)
   total += labels.to(device, dtype=torch.long).size(0)
   correct += (predicted.to(device) == labels.to(device, dtype=torch.long)).sum()
   correct_lst.append(correct.item() / total)
   for i in range(len(images)):
       preds.append(predicted[i])
       labs.append(labels[i])

# testing accuracy plot
plt.figure(2)
plt.plot(np.array(correct_lst))
plt.ylabel("Accuracy")
plt.xlabel("Iterations")
plt.title("Test Accuracy")
##plt.show(block=False)
plt.show()
##plt.savefig("Training_Loss_vs_Epochs.png")

print(60*"-")
print('Test Accuracy of the model on the 126,084 test images: %d %%' % (100 * correct / total))
print(60*"-")

########################################################################################################################

## confusion matrix
cm = confusion_matrix(labs, preds)
print("The confusion matrix is :\n " +str(cm))
print("")

print("\nThe f1 score is : %.3f" % (f1_score(labs, preds, average="macro")))

print("\nThe accuracy score is : %.3f " % (accuracy_score(labs, preds, normalize=True)))

print("\nThe misclassification rate is : %.3f " % (1 - (accuracy_score(labs, preds, normalize=True))))

print("")
target_names = ['label 0', 'label 1', 'label 2', 'label 3', 'label 4', 'label 5', 'label 6', 'label 7', 'label 8', 'label 9']
print(classification_report(torch.FloatTensor(labs), torch.FloatTensor(preds), target_names=target_names))

########################################################################################################################

end = time.time()
print("\nThe execution time is: %.2f " % (end - start) + " seconds")

os.chdir("..") # changing back to the working directory
print("\nThe current working directory is: " +str(os.getcwd()))

#########################################################  END  ########################################################
