import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#print(x_train.shape)
#print(x_train[0].shape)
x_train = x_train/255.0
x_test = x_test/255.0


model = Sequential()

model.add(LSTM(128, input_shape=(x_train.shape[1:]), activation='tanh', recurrent_dropout=0, unroll=False, use_bias=True, return_sequences=True)) #CuDNNLSTM
model.add(Dropout(0.2))

model.add(LSTM(128, activation='tanh', recurrent_dropout=0, unroll=False, use_bias=True)) #CuDNNLSTM
model.add(Dropout(0.2))

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(10, activation='softmax'))    

opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-6)

model.compile(loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy'])

model.fit(x_train,
    y_train,
    epochs=3,
    validation_data=(x_test, y_test))