import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
# more info on callbakcs: https://keras.io/callbacks/ model saver is cool too.
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import pickle
import time

NAME = "Cats-vs-dogs-CNN-64x2-{}".format(int(time.time()))

tensorboard = TensorBoard(log_dir = 'logs/{}'.format(NAME))

pickle_in = open("X.pickle","rb")
X = np.array(pickle.load(pickle_in))
X = X/255.0

pickle_in = open("y.pickle","rb")
y = np.array(pickle.load(pickle_in))

model = Sequential()

dense_layers = [0]
layer_sizes = [64]
conv_layers = [3]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = "{}-Conv-{}-Nodes-{}-Dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
            
            model.add(Conv2D(64, (3, 3), input_shape=X.shape[1:]))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            for l in range(conv_layer - 1):
                model.add(Conv2D(64, (3, 3)))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Flatten())

            for _ in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation('relu'))
        
            model.add(Dense(1))
            model.add(Activation('sigmoid'))

            model.compile(loss = 'binary_crossentropy',
                          optimizer = 'adam',
                          metrics = ['accuracy'])

            model.fit(X, y,
                      batch_size = 32,
                      epochs = 1,
                      validation_split = 0.3,
                      callbacks = [tensorboard])
