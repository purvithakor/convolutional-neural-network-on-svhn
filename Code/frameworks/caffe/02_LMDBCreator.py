import warnings
warnings.filterwarnings('ignore')
import os
import random
import numpy as np
import fnmatch
import cv2
import caffe
import lmdb
import wget
from scipy.io import loadmat
import numpy as np
from PIL import Image
#DOWNLOADING PROCESSED MATRIX FILES FROM BUCKET INTO SVHN_DATA

print(os.getcwd()) #/home/ubuntu/Deep-Learning/Project_Caffe_testing
os.chdir('./svhn_data')
filename_list = ['test_processed.mat', 'train_processed.mat']

for filename in filename_list:
    filepath = './'
    if os.path.exists(filepath):
        if not os.path.isfile(filename):
                print 'Downloading ' + filename + ' file'
                url = 'https://storage.googleapis.com/1_deep_learning_final_project_group_1/processed_files/'+ filename
                wget.download(url, filepath)
                print('Downloaded')
        else:
            print 'file ' + filename + ' already exist.'
    else:
        print 'folder ' + filename + ' already exist.'

# ################CHECK FOLDER EXIST FOR TRAIN

folder_name = 'train_images'

print 'Checking if ', folder_name , ' directory exist...\n'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
    os.chmod(folder_name, 0o777)
    print 'Directory does not exist. Creating ' , folder_name , ' directory now...'
    print 'Directory ', folder_name, ' created'
else:
    print 'Directory' , folder_name, ' already exists'

# ################CHECK FOLDER EXIST FOR TEST

folder_name = 'test_images'

print 'Checking if ', folder_name , ' directory exist...\n'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
    os.chmod(folder_name, 0o777)
    print 'Directory does not exist. Creating ' , folder_name , ' directory now...\n'
    print 'Directory ', folder_name, ' created'
else:
    print 'Directory' , folder_name, ' already exists'

# ################CHECK IF LABEL FOLDERS EXIST IN TEST ( COMMENTING TO SAVE TIME)

os.chdir('./test_images/')

label_list = ['label_0','label_1','label_2','label_3','label_4','label_5','label_6','label_7','label_8','label_9']

for folder in label_list:
    filepath = os.getcwd() + "/" + folder
    if not os.path.exists(filepath):
        os.makedirs(filepath)
        os.chmod(folder, 0o777)
        print 'Directory does not exist. Creating ', folder, ' directory now...'
        print 'Directory ', folder, ' Created'
    else:
        print 'Directory ' + folder + ' already exists.'

# ################CHECK IF LABEL FOLDERS EXIST IN TRAIN ( COMMENTING TO SAVE TIME)

os.chdir("..")
os.chdir('./train_images/')

for folder in label_list:
    filepath = os.getcwd() + "/" + folder
    if not os.path.exists(filepath):
        os.makedirs(filepath)
        os.chmod(folder, 0o777)
        print 'Directory does not exist. Creating ', folder, ' directory now...'
        print 'Directory ', folder, ' Created'
    else:
        print 'Directory ' + folder + ' already exists.'

os.chdir("../..")
#######################################################################################################################

# separate X and Y
train_data = loadmat('./svhn_data/train_processed.mat')
test_data = loadmat('./svhn_data/test_processed.mat')

x_test = np.array(test_data['x_test'])
y_test = np.array(test_data['y_test'])

x_train = np.array(train_data['x_train'])
y_train = np.array(train_data['y_train'])

########################################################################################################################

#print os.getcwd()

# CONVERTING TEST MAT TO IMAGES
#txt_test_file = open("./svhn_data/test.txt", "w")
tot_test_img = len(x_test[1,1,1,:])

for i in range(tot_test_img) :
    label = y_test[i].item()
    Image.fromarray(x_test[:,:,:,i]).save('./svhn_data/test_images/label_'+str(label)+'/image'+str(i)+'.jpg')
#    txt_test_file.write("test_images/label_"+str(label)+"/image"+str(i)+".jpg="+str(label)+"\n")
#txt_test_file.close()

# CONVERTING TRAIN MAT TO IMAGES
#print os.getcwd()
#txt_train_file = open("./train.txt", "w")
tot_train_img = len(x_train[1,1,1,:])

for i in range(tot_train_img) :
    label = y_train[i].item()
    Image.fromarray(x_train[:,:,:,i]).save('./svhn_data/train_images/label_'+str(label)+'/image'+str(i)+'.jpg')
#    txt_train_file.write("train_images/label_"+str(label)+"/image"+str(i)+".jpg="+str(label)+"\n")
#txt_train_file.close()

#################################################################################
### Things you need to change before using this script on your own image data ###
#################################################################################

# 3. Put the path that you want to write lmdb files into
# !!!Do not put exist folder here because it will delete these folder every time you run the script

# 4. Put the size you want your images resize to
IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32

# You are ready to run this script now
# ====================================================================================

def make_datum(img, label):
    return caffe.proto.caffe_pb2.Datum(
        channels=3,
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        label=label,
        data=np.rollaxis(img, axis=2, start=0).tostring())

train_lmdb = 'train_lmdb_db'
test_lmdb = 'test_lmdb_db'

# 4. Put training jpg images path here
JPG_train_path = './svhn_data/train_images'

# Put test jpg images path here
JPG_test_path = './svhn_data/test_images'

#os.system('rm -rf  ' + train_lmdb)
#os.system('rm -rf  ' + test_lmdb)

# universal way
# a list to store all images' path
i = 0
train_data = []
for root, dirnames, filenames in os.walk(JPG_train_path):
    i = i + 1
    for filename in fnmatch.filter(filenames, '*.jpg'):
        train_data.append(os.path.join(root,filename))

num_train = len(train_data)
num_label_train = i - 1

k = 0
test_data = []
for root, dirnames, filenames in os.walk(JPG_test_path):
    k = k + 1
    for filename in fnmatch.filter(filenames, '*.jpg'):
        test_data.append(os.path.join(root,filename))
num_test = len(test_data)
num_label_test = k - 1

print '\nCreating train_lmdb'
env_db = lmdb.open(train_lmdb, map_size=int(1e12))
with env_db.begin(write=True) as txn:
    for idx, img_path in enumerate(train_data):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        label = int(img_path.split('/')[-2][-1])
        datum = make_datum(img, label)
        txn.put('{:0>5d}'.format(idx), datum.SerializeToString())
        print '{:0>5d}'.format(idx) + ':' + img_path
env_db.close()

print '\nCreating test_lmdb'
env_db = lmdb.open(test_lmdb, map_size=int(1e12))
with env_db.begin(write=True) as txn:
    for idx, img_path in enumerate(test_data):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        label = int(img_path.split('/')[-2][-1])
        datum = make_datum(img, label)
        txn.put('{:0>5d}'.format(idx), datum.SerializeToString())
        print '{:0>5d}'.format(idx) + ':' + img_path
env_db.close()

print '\nFinished processing all images'
print '\nTraining data has {} images in {} labels'.format(num_train, num_label_train)
print '\nTesting data has {} images in {} labels'.format(num_test, num_label_test)

print "Execute 03_SVHNCaffe.py now!"
