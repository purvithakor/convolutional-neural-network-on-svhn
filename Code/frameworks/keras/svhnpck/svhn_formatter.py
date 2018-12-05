import numpy as np

# ---------------------Setting up Images and Labels-----------------------

def one_hot_encoder(y):  # function will one hot encode the labels
    y = (np.arange(10) == y[:, np.newaxis]).astype(np.float32)
    return y

# change the dimensions of the image data that is needed to tensorflow and Keras
def change_dim_x(x_data):
    print('\n\t\tOLD DIMENSIONS\t\t\t\t\t  NEW DIMENSIONS')
    print('[H_dim, W_dim, Chan, numImg] ---->[numImg, H_dim, W_dim, Chan]')

    new_x = (x_data.transpose(3, 0, 1, 2)/255).astype(np.float32)
    print('\t' + str(x_data.shape) + '\t\t\t\t\t' + str(new_x.shape) + '\n')
    return new_x

# changes the amount of data for testing purposesself.
# For example, I used only 100,000 training images to make the program to run the program.
def change_range(num, x_array, y_array):
    x_array = x_array[:num, ]
    y_array = y_array[:num, ]
    print('x_train shape: ' + str(x_array.shape) + '\n')
    print('y_train shape: ' + str(y_array.shape) + '\n')

    return x_array, y_array


if __name__ == '__main__':
    one_hot_encoder()
    change_dim_x()
    change_range()
