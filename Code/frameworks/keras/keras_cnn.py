"""
GROUP 1: Cesar Lopez, Akshay Kamath, Purvi Thakor.

Built a simple Keras Convolutions Neural network models for the SVHN datasetself.

"""
from svhnpck.data_loader import SvhnData
from svhnpck.svhn_formatter import change_range
from keras import layers
from keras import models
from keras.callbacks import TensorBoard
from keras.models import load_model

batch_size = 500

tensorboard = TensorBoard(log_dir='./my_log_dir', histogram_freq=1, write_graph=True, write_images=True)

svhn = SvhnData()  # create an object of the class
svhn.load_data()  # load the data from the bucket
x_train, y_train, x_test, y_test = svhn.get_data(change_dim=True, one_hot=True)  # get the train and test data

x_train, y_train = change_range(300000, x_train, y_train)
x_test, y_test = change_range(40000, x_test, y_test)

print('x_train shape: ' + str(x_train.shape) + '\n')
print('y_train shape: ' + str(y_train.shape) + '\n')

print('x_test shape: ' + str(x_test.shape) + '\n')
print('y_test shape: ' + str(y_test.shape) + '\n')

# Initializing a model
model = models.Sequential()
# adding the first convolution layer
model.add(layers.Conv2D(16, (5, 5), activation='relu', input_shape=(32, 32, 3)))
# Adding a max pooling layer
model.add(layers.MaxPooling2D((2, 2)))

# adding the second convolution and max pooling layer
model.add(layers.Conv2D(32, (5, 5), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=batch_size, epochs=10, callbacks= [tensorboard])

# model.save('./keras_models/all_data_2Convo_5x5_dp2.h5')
# print('\nModel has been saved')
print('\n')
print(model.summary())

# Type this on Terminal in Same Directory as your Keras code
# tensorboard --logdir ./my_log_dir/
