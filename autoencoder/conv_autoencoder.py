from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras.datasets import fashion_mnist
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.callbacks import TensorBoard
import tensorflow_core.python
import tensorflow
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import os
from PIL import Image



# get images
def getAndPrepareImages(path) :
    print('data prep begins')
    img_array = []
    for filename in os.listdir(path):
        temp = cv2.imread(path + filename)
        #temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
        temp = cv2.resize(temp, (128, 128))
        img_array.append(temp)

    img_array = np.array(img_array, dtype="float")

    return img_array

# train
def train_autoencoder():
    print('training begins')

    #######################################################

    X_train = getAndPrepareImages('C:\\Users\\tim.reicheneder\\Desktop\\Bachelorthesis\\impl_final\\pictures\\')
    X_test = X_train
    #######################################################
    # X_train = X_train.astype('float32')/255
    # X_test = X_test.astype('float32')/255
    # X_train = X_train.reshape(len(X_train), np.prod(X_train.shape[1:]))
    # X_test = X_test.reshape(len(X_test), np.prod(X_test.shape[1:]))
    print(X_train.shape)
    print(X_test.shape)


    input_img = Input(shape=(128, 128, 3))
    encoded = Conv2D(32, (3, 3), activation='relu')(input_img)
    encoded = MaxPooling2D((2, 2))(encoded)
    encoded = Conv2D(64, (3, 3), activation='relu')(encoded)
    encoded = MaxPooling2D((2, 2))(encoded)
    encoded = Conv2D(128, (3, 3), activation='relu')(encoded)
    encoded = MaxPooling2D((2, 2))(encoded)
    encoded = Conv2D(256, (3, 3), activation='relu')(encoded)

    decoded = Conv2D(256, (3, 3), activation='relu')(encoded)
    decoded = UpSampling2D((2, 2))(decoded)
    decoded = Conv2D(128, (3, 3), activation='relu')(encoded)
    decoded = UpSampling2D((2, 2))(decoded)
    decoded = Conv2D(64, (3, 3), activation='relu')(encoded)
    decoded = UpSampling2D((2, 2))(decoded)
    decoded = Conv2D(1, (3, 3), activation='sigmoid')(encoded)

    autoencoder=Model(input_img, decoded)

    encoder = Model(input_img, encoded)

    autoencoder.summary()

    encoder.summary()

    autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    autoencoder.fit(X_train, X_train,
        epochs=1000,
        batch_size=256,
        shuffle=True,
        validation_data=(X_test, X_test))

    encoded_imgs = encoder.predict(X_test)
    predicted = autoencoder.predict(X_test)

    # save model
    autoencoder.save('.\model')

    for i in range(10):
        # display original images
        ax = plt.subplot(2, 10, i + 1)
        plt.imshow(X_test[i].reshape(128, 128))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        # # display encoded images
        # ax = plt.subplot(3, 20, i + 1 + 20)
        # plt.imshow(encoded_imgs[i].reshape(8,4))
        # plt.gray()
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)
        # display reconstructed images
        ax = plt.subplot(2, 10, 10 + i + 1)
        plt.imshow(predicted[i].reshape(128, 128))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
  
    plt.show()

if __name__ == "__main__":
    train_autoencoder()