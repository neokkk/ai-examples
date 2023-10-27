import cv2 as cv
import math
import numpy as np
from tensorflow import keras
from keras.callbacks import LearningRateScheduler
from keras.datasets import cifar10
from keras.initializers import Constant, glorot_uniform, zeros
from keras.layers import AveragePooling2D, Conv2D, Dense, Dropout, Flatten, GlobalAveragePooling2D, Input, MaxPooling2D, concatenate
from keras.models import Model
from keras.optimizers import SGD
from keras.utils import to_categorical

class GoogLeNet:
  def __init__(self, x_train, y_train, x_test, y_test):
    self.x_train = x_train
    self.y_train = y_train
    
  def build_model(self, kernel_init, bias_init, img_input):
    # Stage 1
    x = Conv2D(64, (7, 7), strides=(2, 2), activation='relu', padding='same', kernel_initializer=kernel_init, bias_initializer=bios_init)(img_input)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = Conv2D(192, (3, 3), strides=(1, 1), activation='relu', padding='same', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    # Stage 2-1
    x = inception_module(x, 64, 96, 128, 16, 32, 32, name='inception_3a', kernel_init=kernel_init, bias_init=bias_init)
    x = inception_module(x, 128, 128, 192, 32, 96, 64, name='inception_3b', kernel_init=kernel_init, bias_init=bias_init)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = inception_module(x, 192, 96, 208, 16, 48, 64, name='inception_4a')
    # Stage 2-a (Auxiliary Classifier)
    x1 = AveragePooling2D((5, 5), strides=(3, 3))(x)
    x1 = Conv2D(128, (1, 1), activation='relu', padding='same')(x1)
    x1 = Flatten()(x1)
    x1 = Dense(1024, activation='relu')(x1)
    x1 = Dense(10, activation='softmax')(x1)
    # Stage 2-2
    x = inception_module(x, 160, 112, 224, 24, 64, 64, name='inception_4b', kernel_init=kernel_init, bias_init=bias_init)
    x = inception_module(x, 128, 128, 256, 24, 64, 64, name='inception_4c', kernel_init=kernel_init, bias_init=bias_init)
    x = inception_module(x, 112, 144, 288, 32, 64, 64, name='inception_4d', kernel_init=kernel_init, bias_init=bias_init)
    # Stage 2-b (Auxiliary Classifier)
    x2 = AveragePooling2D((5, 5), strides=(3, 3))(x)
    x2 = Conv2D(128, (1, 1), activation='relu', padding='same')(x2)
    x2 = Flatten()(x2)
    x2 = Dense(1024, activation='relu')(x2)
    x2 = Dense(10, activation='softmax')(x2)
    # Stage 2-3
    x = inception_module(x, 256, 160, 320, 32, 128, 128, name='inception_4e', kernel_init=kernel_init, bias_init=bias_init)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = inception_module(x, 256, 160, 320, 32, 128, 128, name='inception_5a', kernel_init=kernel_init, bias_init=bias_init)
    x = inception_module(x, 384, 192, 384, 48, 128, 128, name='inception_5b', kernel_init=kernel_init, bias_init=bias_init)
    # Stage 3
    x = GlobalAveragePooling2D()(x) # 1x1
    x = Dropout(0.4)(x)
    x = Dense(10, activation='softmax')(x)
    
    self.model = Model(img_input, [x, x1, x2], name='googlenet')
    
  def inception_module(x,
                       filters_1x1,
                       filters_3x3_reduce,
                       filters_3x3,
                       filters_5x5_reduce,
                       filters_5x5,
                       filters_pool_proj,
                       name=None,
                       kernel_init='glorot_uniform',
                       bias_init='zeros'):
    conv_1x1 = Conv2D(filters_1x1, (1, 1), activation='relu', padding='same', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    conv_3x3_reduce = Conv2D(filters_3x3_reduce, (1, 1), activation='relu', padding='same', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    conv_3x3 = Conv2D(filters_3x3, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_init, bias_initializer=bias_init)(conv_3x3_reduce)
    conv_5x5_reduce = Conv2D(filters_5x5_reduce, (1, 1), activation='relu', padding='same', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    conv_5x5 = Conv2D(filters_5x5, (5, 5), activation='relu', padding='same', kernel_initializer=kernel_init, bias_initializer=bias_init)(conv_5x5_reduce)
    max_pool = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    pool_proj = Conv2D(filters_pool_proj, (1, 1), activation='relu', padding='same', kernel_initializer=kernel_init, bias_initializer=bias_init)(max_pool)
    
    output = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3, name=name)
    return output
  
  def decay(epoch, steps=100):
    initial_lrate = 0.01
    drop = 0.96
    epochs_drop = 8
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate
  
  def train(self):
    sgd = SGD(lr=0.01, momentum=0.9, nestrov=False)
    lr_scheduler = LearningRateScheduler(decay, verbose=1)
    
    self.model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'],
                       loss_weight=[1, 0.3, 0.3],
                       optimizer=sgd,
                       metrics=['accuracy'])
    self.model.fit(self.x_train, self.y_train, epochs=10, batch_size=128, verbose=1, callbacks=[lr_scheduler])
    self.model.save('googlenet.h5')
    
  def evaluate(self):
    scores = self.model.evaluate(self.x_test, self.y_test)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
    
def load_cifar10_data():
  (x_train, y_train), (x_test, y_test) = cifar10.load_data()
  return (x_train, y_train), (x_test, y_test)

def run_GoogLeNet():
  (x_train, y_train), (x_test, y_test) = load_cifar10_data()
  kernel_init = glorot_uniform()
  bias_init = Constant(value=0.2)
  img_input = Input(shape=(224, 224, 3))
  model = GoogLeNet(x_train, y_train, x_test, y_test)
  model.build_model(kernel_init, bias_init, img_input)
  model.train()
  model.evaluate()