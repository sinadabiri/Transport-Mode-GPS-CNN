import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
import pickle
from keras.optimizers import Adam
import random
from sklearn.model_selection import train_test_split
import time
import tensorflow as tf
from sklearn.metrics import classification_report
from keras.callbacks import ModelCheckpoint
tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
sess = print(tf.Session(config=tf.ConfigProto(log_device_placement=True)))

start_time = time.clock()
np.random.seed(7)
random.seed(7)

filename = '../Combined Trajectory_Label_Geolife/Revised_KerasData_Smoothing.pickle'
with open(filename, mode='rb') as f:
    TotalInput, FinalLabel = pickle.load(f, encoding='latin1')  # Also can use the encoding 'iso-8859-1'

NoClass = len(list(set(np.ndarray.flatten(FinalLabel))))
Threshold = len(TotalInput[0, 0, :, 0])

# Making training and test data: 80% Training, 20% Test
Train_X, Test_X, Train_Y, Test_Y_ori = train_test_split(TotalInput, FinalLabel, test_size=0.20, random_state=7)

Train_Y = keras.utils.to_categorical(Train_Y, num_classes=NoClass)
Test_Y = keras.utils.to_categorical(Test_Y_ori, num_classes=NoClass)

# Model and Compile
model = Sequential()
activ = 'relu'
model.add(Conv2D(32, (1, 3), strides=(1, 1), padding='same', activation=activ, input_shape=(1, Threshold, 4)))
model.add(Conv2D(32, (1, 3), strides=(1, 1), padding='same', activation=activ))

model.add(Flatten())
model.add(Dense(NoClass, activation='softmax'))

optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
# checkpoint
filepath="CNN-A-weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
offline_history = model.fit(Train_X, Train_Y, epochs=62, batch_size=64, shuffle=False,
                            validation_data=(Test_X, Test_Y), callbacks=callbacks_list)
hist = offline_history
A = np.argmax(hist.history['val_acc'])
print('the optimal epoch size: {}, the value of high accuracy {}'.format(hist.epoch[A], np.max(hist.history['val_acc'])))


# Calculating the test accuracy, precision, recall
# Prediction
# load weights
model.load_weights("CNN-A-weights.best.hdf5")
# Compile model (required to make predictions)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
Pred = model.predict(Test_X, batch_size=32)
Pred_Label = np.argmax(Pred, axis=1)

# Precision, recall, F-score
print(classification_report(Test_Y_ori, Pred_Label, digits=3))
# Report the accuracy at the final
print('Optimal Accurcay: ', model.evaluate(x=Test_X, y=Test_Y, batch_size=64))
print(time.clock() - start_time, "seconds")
