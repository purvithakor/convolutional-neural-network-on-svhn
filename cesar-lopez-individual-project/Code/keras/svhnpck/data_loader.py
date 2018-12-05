"""
CLASS to get training and Testing data from our google bucket
This is used as a package. It will be called in at least the tensorflow and
keras.
"""
import scipy.io as sio  # To load the matlabfiles
import wget  # Get the data from url
import os
from svhnpck.svhn_formatter import change_dim_x, one_hot_encoder  # functions from other files

class SvhnData:

    def __init__(self):
        self.folder_name = 'svhn_data'
        self.filename_list = ['test_processed.mat', 'train_processed.mat']

        self.train_data = None
        self.test_data = None

        self.x_train = None
        self.y_train = None

        self.x_test = None
        self.y_test = None

    # Checking if svhn_data folder exist. if not, then create it
    def load_data(self):
        print('Checking if ', self.folder_name, ' directory exist...\n')
        if not os.path.exists(self.folder_name):
            os.makedirs(self.folder_name)
            print('Directory does not exist. Creating ', self.folder_name, ' directory now...\n')
            print('Directory ', self.folder_name, ' Created')
        else:
            print('Directory', self.folder_name, 'already exists')

        print('downloading svhn data files...')

        # ----------------DOWNLOADING PROCESSED MATRIX FILES FROM BUCKET-------------------
        # Checking to see if the data files exist. If not, then download them
        for filename in self.filename_list:
            filepath = './svhn_data/' + filename
            if not os.path.exists(filepath):
                print('\nDownloading ' + filename + ' file')
                url = 'https://storage.googleapis.com/1_deep_learning_final_project_group_1/processed_files/' + filename
                wget.download(url, filepath)
                print(' Downloaded')
            else:
                print('file ' + filename + ' already exist.')

        print('\n' + '*' * 10 + 'Downloading Done' + '*' * 10 + '\n')

    # Setting the train and testing images and labels to variables.
    @staticmethod
    def get_data(change_dim=False, one_hot=False):

        train_data = sio.loadmat('svhn_data/train_processed.mat')
        test_data = sio.loadmat('svhn_data/test_processed.mat')

        x_train = train_data['x_train']
        y_train = train_data['y_train']

        x_test = test_data['x_test']
        y_test = test_data['y_test']

        # Changing the dimensions to the way tensorflow likes it.
        if change_dim:
            x_train = change_dim_x(x_train)
            x_test = change_dim_x(x_test)

        # One_hot_encoder for labels
        if one_hot:
            y_train = y_train.flatten()
            y_train = one_hot_encoder(y_train)

            y_test = y_test.flatten()
            y_test = one_hot_encoder(y_test)

        print('x_train shape: ' + str(x_train.shape) + '\n')
        print('y_train shape: ' + str(y_train.shape) + '\n')

        print('x_test shape: ' + str(x_test.shape) + '\n')
        print('y_test shape: ' + str(y_test.shape) + '\n')

        return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    SvhnData()
