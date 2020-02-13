from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras.datasets import fashion_mnist
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.callbacks import TensorBoard
from keras.models import load_model
import tensorflow_core.python
import tensorflow
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import os
from PIL import Image
import autoencoder



autoencoder_model = load_model('.\model')

imgs = autoencoder.getAndPrepareImages('C:\\Users\\tim.reicheneder\\Desktop\\Bachelorthesis\\impl_final\\original_pictures\\')
imgs = imgs.astype('float32')/255
imgs = imgs.reshape(len(imgs), np.prod(imgs.shape[1:]))

predicted = autoencoder_model.predict(imgs)

for i in range(100):
    resize = cv2.resize(predicted[i].reshape(128, 128), (800, 600), cv2.INTER_AREA)
    cv2.imshow("a", resize)
    cv2.resizeWindow("a", (800, 600))
    cv2.waitKey(0)