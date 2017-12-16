from scipy.misc import imresize
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.utils import np_utils, generic_utils
from keras.optimizers import Adam, SGD, RMSprop
from spatial_transformer import SpatialTransformer
import sys

import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
import os

plt.switch_backend('agg')
np.random.seed(1337)  # for reproducibility
batch_size = 256
nb_classes = 10
nb_epochs = 50
restore = False if sys.argv[1] == 'train' else True

DIM = 60
mnist_cluttered = "../datasets/mnist_cluttered_60x60_6distortions.npz"
print("Loading data...")
data = np.load(mnist_cluttered)
X_train, y_train = data['x_train'], np.argmax(data['y_train'], axis=-1)
X_valid, y_valid = data['x_valid'], np.argmax(data['y_valid'], axis=-1) 
X_test, y_test = data['x_test'], np.argmax(data['y_test'], axis=-1)

# reshape for convolutions
X_train = X_train.reshape((X_train.shape[0], DIM, DIM, 1))
X_valid = X_valid.reshape((X_valid.shape[0], DIM, DIM, 1))
X_test = X_test.reshape((X_test.shape[0], DIM, DIM, 1))

y_train = np_utils.to_categorical(y_train, nb_classes)
y_valid = np_utils.to_categorical(y_valid, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)

X_train, y_train = np.concatenate([X_train, X_valid], axis=0), np.concatenate([y_train, y_valid], axis=0)

print("Train samples: {}".format(X_train.shape))
# print("Validation samples: {}".format(X_valid.shape))
print("Test samples: {}".format(X_test.shape))


input_shape = np.squeeze(X_train.shape[1:])
input_shape = (60, 60, 1)
print("Input shape:", input_shape)

# initial weights
b = np.zeros((2, 3), dtype='float32')
b[0, 0] = 1
b[1, 1] = 1
W = np.zeros((50, 6), dtype='float32')
weights = [W, b.flatten()]

print("Building model...")
locnet = Sequential()
locnet.add(MaxPooling2D(pool_size=(2, 2), input_shape=input_shape))
locnet.add(Convolution2D(20, (5, 5)))
locnet.add(MaxPooling2D(pool_size=(2, 2)))
locnet.add(Convolution2D(20, (5, 5)))

locnet.add(Flatten())
locnet.add(Dense(50))
locnet.add(Activation('relu'))
locnet.add(Dense(6, weights=weights))

model = Sequential()

model.add(SpatialTransformer(localization_net=locnet,
                             output_size=(30, 30), input_shape=input_shape))

model.add(Convolution2D(32, (3, 3), padding='same', use_bias=False))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(32, (3, 3), use_bias=False))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))

model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.001, clipnorm=500.))
model.summary()

# for visualize
XX = model.input
YY = model.layers[0].output
F = K.function([XX], [YY])

if os.path.isfile("weights.h5") and restore:
    model.load_weights("weights.h5")
    print("Loaded weights!")

def write_vis(idx, fig):
    X_vis = F([X_test[:9]])
    plt.clf()
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        image = np.squeeze(X_vis[0][i])
        plt.imshow(image, cmap='gray')
        plt.axis('off')
    fig.canvas.draw()
    fig.savefig("../images/%d.png" % idx)

def train():    
    fig = plt.figure()
    num_batches_per_ep = X_train.shape[0]/batch_size + 1
    best_test_score = 1.0
    early_stop = 0
    with open("new_train_log.txt", "w") as log_file:
        try:
            for e in range(nb_epochs):
                print('-' * 40)
                for b in range(num_batches_per_ep):
                    f = b * batch_size
                    l = (b + 1) * batch_size
                    X_batch = X_train[f:l].astype('float32')
                    y_batch = y_train[f:l].astype('float32')
                    loss = model.train_on_batch(X_batch, y_batch)
                    log_file.write("Epoch: %d | Batch: %d | Loss: %f\n" % (e, b, loss))
                scoret = model.evaluate(X_test, y_test, verbose=1)
                if scoret < best_test_score:
                    best_test_score = scoret
                    model.save_weights("weights.h5")
                else:
                    early_stop += 1
                if early_stop > 30:
                    print("\nStop training after 20 non-improved epochs!")
                    break
                print('\nEpoch: {0} | Test: {1}'.format(e, scoret))
                log_file.write('Epoch: {0} | Test: {1}'.format(e, scoret))

        except KeyboardInterrupt:
            pass

def evaluate():
    other_predictions = model.predict(X_test, batch_size=256, verbose=1)
    other_predictions = np.argmax(other_predictions, 1)
    print ("Accuracy on test: %f" % (sum((other_predictions == np.argmax(y_test, 1)))/ 10000.))

def test():
    while 1:
        idx = int(raw_input("Image: "))
        im = X_test[idx:idx+1]
        pred = model.predict(im)
        pred = np.argmax(pred, 1)
        fig = plt.figure()
        fig.suptitle('Prediction: %d | Ground truth: %d' % (pred, np.argmax(y_test[idx:idx+1], 1)), fontsize=14, fontweight='bold')
        ax = fig.add_subplot(121)
        fig.subplots_adjust(top=0.85)
        plt.imshow(np.squeeze(im), cmap='gray')
        ax.set_title('Original')
        ax = fig.add_subplot(122)
        fig.subplots_adjust(top=0.85)
        plt.imshow(np.squeeze(F([im])), cmap='gray')
        ax.set_title('Transformer')
        plt.show()

if __name__ == "__main__":
    if sys.argv[1] == 'train':
        train()
    if sys.argv[1] == 'test':
        test()
    if sys.argv[1] == 'eval':
        evaluate()
