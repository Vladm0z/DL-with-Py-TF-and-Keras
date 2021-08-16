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
#gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.9)
#sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))


pickle_in = open("X.pickle","rb")
A = pickle.load(pickle_in)
X = np.array(A)

pickle_in = open("y.pickle","rb")
B = pickle.load(pickle_in)
y = np.array(B)

X = X/255.0

model = Sequential()

model.add(Conv2D(64, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#model.add(Dense(64))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'],
              )

model.fit(X, y,
          batch_size=32,
          epochs=20,
          validation_split=0.3,
          callbacks=[tensorboard])
