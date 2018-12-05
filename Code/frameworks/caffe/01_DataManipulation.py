import warnings
warnings.filterwarnings('ignore')
import numpy as np
import scipy.io as sio
import seaborn as sns
import matplotlib.pyplot as plt
import wget
import os
from random import shuffle
import cv2
from PIL import Image

sns.set_style("white")

# ----------------------Downloading DATA--------------------------

folder_name = 'svhn_data'
filename_list = ['train_32x32.mat', 'test_32x32.mat', 'extra_32x32.mat']

print('\n')
print('Checking if ' + folder_name + ' directory exists')
print('\n')

if not os.path.exists(folder_name):
    print('Directory does not exist. Creating ' + folder_name + ' directory now')
    print('\n')
    os.mkdir(folder_name)
    print('Directory ' + folder_name + ' created')

else:
    print('Directory ' + folder_name + ' already exists.')

print('\n')
print('Downloading svhn data files...')
print('\n')

for filename in filename_list:
    filepath = './svhn_data/' + filename
    if not os.path.exists(filepath):
        print('Downloading ' + filename + ' file')
        print('\n')
        url = 'http://ufldl.stanford.edu/housenumbers/' + filename
        wget.download(url, filepath)
    else:
        print('File ' + filename + ' already exists.')
        print('\n')
print(20*"+")
print('Downloading done')

# ------------------------------------------------------------------------

def image_compare(img,lab,fig_name):
    plt.figure(str(fig_name))
    for i in range(1, 10):
        plt.subplot(3, 3, i)
        plt.imshow(img[:,:,:,i])
        plt.title('Num ' + str(lab[i]))
        plt.xticks()
        plt.yticks()
    plt.tight_layout()
    plt.show(block=False)
    return

# ---------------------------LOADING SVHN DATA----------------------------

# These file contains dictionaries.
# The dictionaries keys are: dict_keys(['y', 'X', '__version__', '__header__', '__globals__'])
# We are only concerned with the 'y' and 'X'.
# The 'y' key contains the labels (What the number is in the image)
# The 'X' key contains the actual images.

train_data = sio.loadmat('svhn_data/train_32x32.mat')
test_data = sio.loadmat('svhn_data/test_32x32.mat')
extra_data = sio.loadmat('svhn_data/extra_32x32.mat')

# Combining X from train, test & extra & stacking them one above the other

x_train = np.array(train_data['X'])
x_test = np.array(test_data['X'])
x_extra = np.array(extra_data['X'])
x = np.concatenate((x_train,x_test,x_extra),axis=-1)

print(20*"+")
print("Combined all image matrices!")

# Combining y from train, test & extra & converting label 10 to 0 across the entire target variable

y_train = train_data['y']
y_test = test_data['y']
y_extra = extra_data['y']
y = np.concatenate((y_train,y_test,y_extra))
y[y == 10] = 0 # label 10 has been converted to 0

print(20*"+")
print("Combined all labels!")

ind_list = [i for i in range(len(x[1,1,1,:]))]
shuffle(ind_list)
x_s = x[:,:,:,ind_list]
y_s = y[ind_list,]

print(20*"+")
print("Data Shuffled!")

# Splitting into train & test

train_pct_index = int(0.8 * (len(x[1,1,1,:])))
X_train, X_test = x_s[:,:,:,:train_pct_index], x_s[:,:,:,train_pct_index:]
y_train, y_test = y_s[:train_pct_index], y_s[train_pct_index:]

#####################################################################

unique1, train_counts = np.unique(y_train, return_counts=True)
train_counts = np.asarray( (unique1, train_counts) ).T
unique2, test_counts = np.unique(y_test, return_counts=True)
test_counts = np.asarray( (unique2, test_counts) ).T

ax1 = plt.subplot(121)
ax1.grid(False)
sns.set_style("white")
sns.barplot(np.arange(0,len(train_counts)),train_counts[:,-1])

plt.xlabel("Categories")
plt.ylabel("Counts")
plt.title("Labels distribution in Train Dataset")

ax2 = plt.subplot(122,sharey=ax1)
ax2.grid(False)
sns.set_style("white")
sns.barplot(np.arange(0,len(test_counts)),test_counts[:,-1])
plt.xlabel("Categories")
plt.ylabel("Counts")
plt.title("Labels distribution in Test Dataset")

plt.show()
#####################################################################
print(20*"+")
print("Data Splitting Completed!")

# PLOTTING IMAGES

# Normalizing images

IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32

def transform_img(img, img_width, img_height):

    #Histogram Equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)

    return img

x_train_normalized = []
x_test_normalized = []

image_compare(X_train,y_train,"before normalizing")

tot_train_images = len(X_train[1,1,1,:])
for i in range(tot_train_images):
    image = X_train[:,:,:,i]
    img = transform_img(image, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
    x_train_normalized.append(img)

x_train_normalized = np.array(x_train_normalized)
x_train_normalized = np.transpose(x_train_normalized,(1,2,3,0))

image_compare(x_train_normalized,y_train,"after normalizing")

print(20*"+")
print("Normalized Training Images!")

tot_test_images = len(X_test[1,1,1,:])
for i in range(tot_test_images):
    image = X_test[:,:,:,i]
    img = transform_img(image, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
    x_test_normalized.append(img)

x_test_normalized = np.array(x_test_normalized)
x_test_normalized = np.transpose(x_test_normalized,(1,2,3,0))

print(20*"+")
print("Normalized Testing Images!")
print(20*"+")
print("Preprocessing Completed!")

# Note - Data has been combined, shuffled, splitted, normalized here. Also label 10 has been converted to 0
# Now we work on Frameworks

#### SHAPE OF X_TRAIN_NORMALIZED IS (32, 32, 3, 504336)
#### SHAPE OF X_TEST_NORMALIZED IS (32, 32, 3, 126084)
#### SHAPE OF Y_TRAIN_NORMALIZED IS (504336, 1)
#### SHAPE OF Y_TEST_NORMALIZED IS (126084, 1)

dict_train = {'x_train':x_train_normalized,'y_train':y_train}
dict_test = {'x_test':x_test_normalized,'y_test':y_test}

sio.savemat('./svhn_data/train_processed.mat',dict_train,format='5')
sio.savemat('./svhn_data/test_processed.mat',dict_test,format='5')

print(20*"+")
print("Files Created!")
print("Execute 02_LMDB_Creator now!")
print(20*"+")

