from tensorflow import keras
from keras.datasets import cifar10
from keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.layers.experimental import preprocessing
from keras.models import Model, Sequential
from keras.utils import to_categorical

class AlexNet:
  def __init__(self, x_train, y_train, x_test, y_test):
    self.x_train = x_train
    self.y_train = y_train
    self.model = Sequential([
      preprocessing.Resizing(227, 227, interpolation='bilinear', input_shape=x_train.shape[1:]),
      # Layer 1
      Conv2D(96, 11, strides=4, activation='relu'),
      BatchNormalization(),
      MaxPooling2D(pool_size=3, strides=2),
      # Layer 2
      Conv2D(256, 5, strides=2, activation='relu', padding='same'),
      BatchNormalization(),
      MaxPooling2D(pool_size=3, strides=2),
      # Layer 3
      Conv2D(384, 3, strides=1, activation='relu', padding='same'),
      BatchNormalization(),
      # Layer 4
      Conv2D(384, 3, strides=1, activation='relu', padding='same'),
      BatchNormalization(),
      # Layer 5
      Conv2D(256, 3, strides=1, activation='relu', padding='same'),
      BatchNormalization(),
      MaxPooling2D(pool_size=3, strides=2),
      # Layer 6, 7, 8 (fully connected)
      Flatten(),
      Dense(4096, activation='relu'),
      Dropout(0.5),
      Dense(4096, activation='relu'),
      Dropout(0.5),
      Dense(10, activation='softmax')
    ])
    
  def train(self):
    self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    self.model.fit(self.x_train, self.y_train, epochs=10, batch_size=128, verbose=1)
    self.model.save('alexnet.h5')
    
  def evaluate(self):
    scores = self.model.evaluate(self.x_test, self.y_test)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

def load_cifar10_data():
  (x_train, y_train), (x_test, y_test) = cifar10.load_data()
  x_train = x_train.astype('float32') / 255
  x_test = x_test.astype('float32') / 255
  y_train = to_categorical(y_train, 10)
  y_test = to_categorical(y_test, 10)
  
  return (x_train, y_train), (x_test, y_test)

def run_AlexNet():
  (x_train, y_train), (x_test, y_test) = load_cifar10_data()
  model = AlexNet(x_train, y_train, x_test, y_test)
  model.train()
  model.evaluate()
  
if __name__ == '__main__':
  run_AlexNet()