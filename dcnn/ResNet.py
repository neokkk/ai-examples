import cv2 as cv
import numpy as np
from tensorflow import keras
from keras.datasets import cifar10
from keras.layers import BatchNormalization, Conv2D, Input, Sequential, activations
from keras.utils import to_categorical

class ResidualUnit(Layer):
  def __init__(self, filters, strides=1, activation='relu', **kwargs):
    super().__init__(**kwargs)
    self.activation = activations.get(activation)
    self.main_layers = [
      Conv2D(filters, 3, strides=strides, padding='same', use_bias=False),
      BatchNormalization(),
      self.activation,
      Conv2D(filters, 3, strides=1, padding='same', use_bias=False),
      BatchNormalization()
    ]
    self.skip_layers = []
    if strides > 1:
      self.skip_layers = [
        Conv2D(filters, 1, strides=strides, padding='same', use_bias=False),
        BatchNormalization()
      ]
  
  def call(self, inputs):
    Z = inputs
    for layer in self.main_layers:
      Z = layer(Z)
    skip_Z = inputs
    for layer in self.skip_layers:
      skip_Z = layer(skip_Z)
    return self.activation(Z + skip_Z)

class ResNet:
  def __init__(self, x_train, y_train, x_test, y_test):
    self.x_train = x_train
    self.y_train = y_train
    self.x_test = x_test
    self.y_test = y_test
  
  def build_model():
    model = Sequential()
    model.add(Conv2D(64, 7, strides=(2, 2), input_shape=[224, 224, 3], padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
    prev_filters = 64
    for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
      strides = 1 if filters == prev_filters else 2
      model.add(ResidualUnit(filters, strides=strides))
      prev_filters = filters
    model.add(GlobalAveragePooling2D())
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    self.model = model
    
  def train(self):
    self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    self.model.fit(self.x_train, self.y_train, epochs=10, batch_size=32, verbose=1)
    
  def evaluate(self):
    scores = self.model.evaluate(self.x_test, self.y_test)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

def load_cifar10_data():
  (x_train, y_train), (x_test, y_test) = cifar10.load_data()
  return (x_train, y_train), (x_test, y_test)

def run_ResNet():
  (x_train, y_train), (x_test, y_test) = load_cifar10_data()
  model = ResNet(x_train, y_train, x_test, y_test)
  model.build_model()
  model.train()
  model.evaluate()