import cv2 as cv
import numpy as np
from tensorflow import keras
from keras.datasets import cifar10
from keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D
from keras.layers.experimental import preprocessing
from keras.models import Model, Sequential
from keras.utils import to_categorical

class VGGNet():
  def __init__(self, x_train, y_train, x_test, y_test, img_input):
    self.x_train = x_train
    self.y_train = y_train
    model = Conv2D(64, 3, activation='relu', padding='same')(img_input)
    model = Conv2D(64, 3, activation='relu', padding='same')(model)
    model = MaxPooling2D(2, strides=2)(model)
    model = Conv2D(128, 3, activation='relu', padding='same')(model)
    model = Conv2D(128, 3, activation='relu', padding='same')(model)
    model = MaxPooling2D(2, strides=2)(model)
    model = Conv2D(256, 3, activation='relu', padding='same')(model)
    model = Conv2D(256, 3, activation='relu', padding='same')(model)
    model = Conv2D(256, 3, activation='relu', padding='same')(model)
    model = MaxPooling2D(2, strides=2)(model)
    model = Conv2D(512, 3, activation='relu', padding='same')(model)
    model = Conv2D(512, 3, activation='relu', padding='same')(model)
    model = Conv2D(512, 3, activation='relu', padding='same')(model)
    model = MaxPooling2D(2, strides=2)(model)
    model = Conv2D(512, 3, activation='relu', padding='same')(model)
    model = Conv2D(512, 3, activation='relu', padding='same')(model)
    model = Conv2D(512, 3, activation='relu', padding='same')(model)
    model = MaxPooling2D(2, strides=2)(model)
    model = Flatten()(model)
    model = Dense(4096, activation='relu')(model)
    model = Dense(4096, activation='relu')(model)
    model = Dense(10, activation='softmax')(model)
    self.model = Model(inputs=img_input, outputs=model, name='vgg16')
    
  def train(self):
    self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    self.model.fit(self.x_train, self.y_train, epochs=10, batch_size=32, verbose=1)
    
  def evaluate(self):
    scores = self.model.evaluate(self.x_test, self.y_test)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

def load_cifar10_data(img_rows, img_cols):
  (x_train, y_train), (x_test, y_test) = cifar10.load_data()
  x_train = x_train[0:2000, :, :, :]
  y_train = y_train[0:2000]
  x_test = x_test[0:500, :, :, :]
  y_test = y_test[0:500]
  x_train = np.array([cv.resize(img, (img_rows, img_cols)) for img in x_train[:, :, :, :]]).astype('float32') / 255
  x_test = np.array([cv.resize(img, (img_rows, img_cols)) for img in x_test[:, :, :, :]]).astype('float32') / 255
  return (x_train, y_train), (x_test, y_test)  

def run_VGGNet16():
  (x_train, y_train), (x_test, y_test) = load_cifar10_data(224, 224)
  img_input = Input(shape=(224, 224, 3))
  model = VGGNet(x_train, y_train, x_test, y_test, img_input)
  model.train()
  model.evaluate()
  
if __name__ == '__main__':
  run_VGGNet16()