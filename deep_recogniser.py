import pandas as pd
import cv2
import numpy as np
'''import keras'''

import matplotlib.pyplot as plt
import matplotlib.cm as cm

np.random.seed(7)

learning_rate = 1e-4
train_iterations = 2000
dropout = 0.5
batch_size = 50
validation_size = 2000
image_to_display = 10
num_labels = 10

data_train = pd.read_table('Kagg/train.csv', delimiter = ',')
data_test = pd.read_table('Kagg/test.csv', delimiter = ',')

images_train = data_train.iloc[:,1:].values
images_test = data_test.values

images_train = images_train.astype(np.float)
images_test = images_test.astype(np.float)

images_train = np.multiply(images_train, 1.0/255.0)
images_test = np.multiply(images_test, 1.0/255.0)

train_labels_flat = data_train[[0]].values.ravel()
# test_labels_flat = data_test[[0]].values.ravel()

train_labels_count = np.unique(train_labels_flat).shape[0]
# test_labels_count = np.unique(test_labels_flat).shape[0]

from keras.utils.np_utils import to_categorical

train_labels = to_categorical(np.asarray(train_labels_flat))
# test_labels = to_categorical(np.asarray(test_labels_flat))
train_labels = train_labels.astype(np.uint8)

images_train = images_train.reshape(images_train.shape[0],28,28)
images_test = images_test.reshape(images_test.shape[0],28,28)

height = 28
width = 28

images_train = images_train.reshape(images_train.shape[0], 1, height, width).astype('float32')
images_test = images_test.reshape(images_test.shape[0], 1, height, width).astype('float32')



from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, LSTM
from keras.optimizers import SGD, Adam
from keras.regularizers import l2

model = Sequential()

model.add(Convolution2D(16, 5, 5, border_mode='valid', input_shape=(1, 28, 28), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(32, 3, 3, activation='relu',border_mode='valid'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(64, 2, 2, activation='relu',border_mode='valid'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(LSTM(128))
# model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(256, activation='relu'))
#model.add(Dropout(0.25))
model.add(Dense(512, activation='relu'))
model.add(Dense(10, activation='softmax'))

# print(model.summary())



adam = Adam(lr = 0.01 , decay = 10**-4)
model.compile(loss = 'categorical_crossentropy', optimizer = adam,
				metrics=['accuracy'])


# model.load_weights('more_accurate.h5')
model.fit(images_train, train_labels, batch_size = 128 , nb_epoch = 10)
# model.fit(images_train, train_labels, batch_size = 128, nb_epoch = 10)
model.save('generator_model.h5')

prediction = model.predict_classes(images_test)

import csv

with open("submit.csv","w", newline="") as out:
    spamwriter = csv.writer(out, delimiter=',')
    spamwriter.writerow(['ImageId','Label'])
    for i in range(len(images_test)):
        spamwriter.writerow([i+1,prediction[i]])

# import csv
# from itertools import izip

# with open('submission.csv', 'wb') as f:
#     writer = csv.writer(f)
#     writer.writerows(izip([i for i in range(len)], frequencies))