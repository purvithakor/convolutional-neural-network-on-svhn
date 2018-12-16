import warnings
warnings.filterwarnings("ignore")
import os
import caffe
import matplotlib.pyplot as plt
import numpy as np
import numpy
import pickle
import wget
import sys
sys.tracebacklimit = 0
import time

start_time_cos = time.time()

np.random.seed(42)
# ##########################IMPORTING DATA FROM BUCKET TO INSTANCE

#print os.getcwd()  # /home/ubuntu/Deep-Learning/Project_Caffe_testing

# ############################CREATING FOLDERS TO SAVE LMDB FILES ON INSTANCE
#
sub_folders = ['train_lmdb_db','test_lmdb_db']

for folder in sub_folders:
    filepath = './' + folder
    if not os.path.exists(filepath):
        os.makedirs(filepath)
        print 'Directory does not exist. Creating ' , folder , ' directory now...\n'
        print 'Directory ', folder, ' Created'
    else:
        print 'Directory', folder, 'already exists'

# ############################DOWNLOADING LMDBs iIN RESPECTIVE FOLDERS

fileList = ['data.mdb', 'lock.mdb']

for directory in sub_folders:

    for filename in fileList:
        filepath = './' + directory + '/' + filename

        if not os.path.exists(filepath):
            print '\nDownloading ' + filename + ' file'
            url = 'https://storage.googleapis.com/1_deep_learning_final_project_group_1/processed_files/' + filepath
            wget.download(url, filepath)
            print(' Downloaded')

        else:
            print('\nfile ' + str(filepath)[-8:] + ' already exists.')

print('\n' + '*' * 10 + 'Downloading Done' + '*' * 10 + '\n\n')

###########################CAFFE EXECUTION

# ADD SUDO CHMOD FOR LMDB

#print os.getcwd()  # /home/ubuntu/Deep-Learning/DL_Project/svhn_data
caffe.set_mode_gpu()

solver = caffe.AdamSolver('svhn_solver.prototxt')
niter = 30000
test_interval = 1000
train_loss = np.zeros(niter)
test_acc = np.zeros(int(np.ceil(niter / test_interval)))

# the main solver loop
for it in range(niter):
    solver.step(1)  # SGD by Caffe
    # store the train loss
    train_loss[it] = solver.net.blobs['loss'].data
    solver.test_nets[0].forward()
    if it % test_interval == 0:
        loss = solver.test_nets[0].blobs['loss'].data
#        print 'Iteration', it, 'testing loss is :',loss
        acc = solver.test_nets[0].blobs['accuracy'].data
        test_acc[it // test_interval] = acc
#        print 'Iteration', it, 'testing accuracy is :',acc

#test_iters = int(126084/ solver.test_nets[0].blobs['data'].num)


# ----------------------------------------------------------------------------------------------
# ##########################Plotting Intermediate Layers, Weight################################
# ---------------------------------------Define Functions---------------------------------------


def vis_square_f(data):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""

    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
                (0, 1), (0, 1))  # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    plt.imshow(data,interpolation='nearest');
    plt.axis('off')
#----------------------------------------------------------------------------------------------
#------------------------------Plot All Feature maps Functions---------------------------------
plt.figure(1)
plt.semilogy(np.arange(niter), train_loss)
plt.xlabel('Number of Iteration')
plt.ylabel('Training Loss Values')
plt.title('Training Loss')
plt.show()

plt.figure(2)
plt.plot(test_interval * np.arange(len(test_acc)), test_acc, 'r')
plt.xlabel('Number of Iteration')
plt.ylabel('Test Accuracy Values')
plt.title('Test Accuracy')
plt.show()

exec_time_cos = time.time()-start_time_cos
print(exec_time_cos)

#----------------------------------------------------------------------------------------------
#------------------------------Plot All Feature maps Functions---------------------------------
# net = solver.net
# f1_0 = net.blobs['conv1'].data[0, :25]
# plt.figure(3)
# vis_square_f(f1_0)
# plt.title('Feature Maps for Conv1')
# plt.show()
# #----------------------------------------------------------------------------------------------
# net = solver.net
# f1_0 = net.blobs['conv2'].data[0, :25]
# plt.figure(4)
# vis_square_f(f1_0)
# plt.title('Feature Maps for Conv2')
# plt.show()
#----------------------------------------------------------------------------------------------
# net = solver.net
# f1_0 = net.blobs['conv3'].data[0, :20]
# plt.figure(5)
# vis_square_f(f1_0)
# plt.title('Feature Maps for Conv3')
# plt.show()

#------------------------------Plot All Kernels for Conv1---------------------------------------
# nrows = 5                                   # Number of Rows
# ncols = 4                                   # Number of Columbs
# ker_size = 5                                # Kernel Size
# Zero_c= np.zeros((ker_size,1))              # Create np.array of zeros
# Zero_r = np.zeros((1,ker_size+1))
# M= np.array([]).reshape(0,ncols*(ker_size+1))
#
# for i in range(nrows):
#     N = np.array([]).reshape((ker_size+1),0)
#
#     for j in range(ncols):
#         All_kernel = net.params['conv1'][0].data[j + i * ncols][0]
#
#         All_kernel = numpy.matrix(All_kernel)
#         All_kernel = np.concatenate((All_kernel,Zero_c),axis=1)
#         All_kernel = np.concatenate((All_kernel, Zero_r), axis=0)
#         N = np.concatenate((N,All_kernel),axis=1)
#     M = np.concatenate((M,N),axis=0)
#
# plt.figure(4)
# plt.imshow(M, cmap='Greys',  interpolation='nearest')
# plt.title('All Kernels for Conv1')
#----------------------------------------------------------------------------------------------
#------------------------------Plot one Kernels for Conv1--------------------------------------
# ker1_0 = net.params['conv1'][0].data[0]      #net.params['conv1'][0] is reffering to Weights
# ker1_0 = numpy.matrix(ker1_0)
# plt.figure(5)
# plt.imshow(ker1_0, cmap='Greys',  interpolation='nearest')
# plt.title('One Kernels for Conv1')
# plt.show()
# #----------------------------------------------------------------------------------------------
#---------------------------Print Shape ans Sizes for all Layers--------------------------------

# for layer_name, blob in net.blobs.iteritems():
#     print layer_name + '\t' + str(blob.data.shape)
#
# for layer_name, param in net.params.iteritems():
#     print layer_name + '\t' + str(param[0].data.shape), str(param[1].data.shape)
