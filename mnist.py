import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.optimizers import SGD

def plot_metric(h, metric):
    train_history = h.history[metric]
    val_history = h.history['val_' + metric]
    epochs = range(1, len(train_history) + 1)
    plt.plot(epochs, train_history)
    plt.plot(epochs, val_history)
    plt.legend(['training ' + metric, 'validation ' + metric])
    plt.title('Training and validation' + metric)
    plt.xlabel('Epochs')
    plt.ylabel(metric)
    plt.show()

def load_mnist_data():
    (train_imgs, train_labels), (test_imgs, test_labels) = mnist.load_data()
    train_imgs, test_imgs = train_imgs / 255.0, test_imgs / 255.0
    return (train_imgs, train_labels), (test_imgs, test_labels)

def build_model():
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='sigmoid'),
        Dense(10, activation='softmax')
    ])
    model.summary()
    model.compile(optimizer=SGD(0.1, momentum=0.9),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def run_mnist():
    (train_imgs, train_labels), (test_imgs, test_labels) = load_mnist_data()
    model = build_model()
    early_stopping = EarlyStopping(monitor='val_loss',
                                   min_delta=0,
                                   patience=10,
                                   verbose=1,
                                   restore_best_weights=True)
 
    hist = model.fit(train_imgs, train_labels, epochs=50, validation_split=0.2, callbacks=[early_stopping])
    plot_metric(hist, 'accuracy')
    plot_metric(hist, 'loss')

    _, train_acc = model.evaluate(train_imgs, train_labels)
    print('훈련 데이터 인식률: ', train_acc)
    _, test_acc = model.evaluate(test_imgs, test_labels)
    print('테스트 데이터 인식률: ', test_acc)

if __name__ == '__main__':
    run_mnist()
