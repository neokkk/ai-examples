import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from keras import Sequential
from keras.datasets import fashion_mnist
from keras.layers import Dense, Flatten, LeakyReLU, Reshape

if __name__ == '__main__':
    (x_train, _), (_, _) = fashion_mnist.load_data()
    x_train = x_train.astype('float') / 255.0

    generator = Sequential([Dense(512, input_shape=[100]),
			    			LeakyReLU(alpha=0.2),
				    		Dense(256),
					    	LeakyReLU(alpha=0.2),
		    				Dense(128),
			    			LeakyReLU(alpha=0.2),
				    		Dense(784),
					    	Reshape([28, 28, 1])])

    discriminator = Sequential([Dense(1, input_shape=[28, 28, 1]),
    							Flatten(),
	    						Dense(256),
		    					LeakyReLU(alpha=0.2),
			    				Dense(128),
				    			LeakyReLU(alpha=0.2),
					    		Dense(64),
						    	LeakyReLU(alpha=0.2),
					    		Dense(1, activation='sigmoid')])

    gan = Sequential([generator, discriminator])
    discriminator.compile(optimizer='adam', loss='binary_crossentropy')
    discriminator.trainable = False
    gan.compile(optimizer='adam', loss='binary_crossentropy')

    batch_size = 64
    noise_shape = 100
    for epoch in range(10):
        for i in range(x_train.shape[0] // batch_size):
            noise = np.random.normal(size=[batch_size, noise_shape])
            gen_img = generator.predict_on_batch(noise)
            train_dataset = x_train[i * batch_size:(i + 1) * batch_size]
            train_label = np.ones(shape=(batch_size, 1)) # real data
            discriminator.trainable = True
            d_loss_real = discriminator.train_on_batch(train_dataset, train_label)
            train_label = np.zeros(shape=(batch_size, 1)) # fake data
            d_loss_fake = discriminator.train_on_batch(gen_img, train_label)
            noise = np.random.normal(size=[batch_size, noise_shape])
            train_label = np.ones(shape=(batch_size, 1))
            discriminator.trainable = False
            d_g_loss_batch = gan.train_on_batch(noise, train_label)

    noise = np.random.normal(size=[10, noise_shape])
    gen_img = generator.predict(noise)
    plt.imshow(noise)
    fig, axe = plt.subplots(2, 5)
    idx = 0
    for i in range(2):
        for j in range(5):
            axe[i, j].imshow(gen_img[idx].reshape(28, 28), cmap='gray')
            idx += 1
