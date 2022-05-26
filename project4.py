from astroNN.datasets import load_galaxy10
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from astroNN.datasets.galaxy10 import galaxy10cls_lookup
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd


np.random.seed(42)
tf.random.set_seed(42)

images, labels = load_galaxy10()

labels = labels.astype(np.float32)
labels = tf.keras.utils.to_categorical(labels, 10) #활성화 함수 적용을 위한 원핫인코딩
images = images.astype(np.float32)
images = images/255 #이미지 정규화

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size = 0.15)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3),activation='relu',
input_shape=(69,69,3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


history = model.fit(X_train, y_train, epochs=20, 
validation_data=(X_test, y_test))

model.evaluate(X_test, y_test, verbose=2)

y_vloss = history.history['val_loss']
y_loss = history.history['loss']

x_len = np.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker='.',c='red', label = 'Testset_loss')
plt.plot(x_len, y_loss, marker='.',c='blue', label = 'Trainset_loss')

plt.legend(loc = 'upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

