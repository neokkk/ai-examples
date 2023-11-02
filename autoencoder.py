import matplotlib.pyplot as plt
from tensorflow import keras
from keras import Sequential
from keras.datasets import mnist
from keras.layers import Dense, Flatten, Reshape
from keras.models import Model

class Autoencoder(Model):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()

        self.encoder = Sequential([
            Flatten(),
            Dense(latent_dim, activation='relu')
        ])
        self.decoder = Sequential([
            Dense(input_dim, activation='sigmoid'),
            Reshape((28, 28))
        ])

    def call(self, input_data):
        encoded_data = self.encoder(input_data)
        decoded_data = self.decoder(encoded_data)
        return decoded_data

if __name__ == '__main__':
    (x_train, _), (x_test, _) = mnist.load_data()

    ae = Autoencoder(784, 64)
    ae.compile(optimizer='adam', loss='mean_squared_error')
    ae.fit(x_train, x_train, epochs=50, batch_size=256, shuffle=True, validation_data=(x_test, x_test))

    encoded_imgs = ae.encoder(x_test)
    decoded_imgs = ae.decoder(encoded_imgs)

    n = 10
    plt.figure(figsize=(10, 2))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
