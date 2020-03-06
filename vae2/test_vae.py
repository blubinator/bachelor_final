import keras
from keras.layers import Conv2D, Conv2DTranspose, Input, Flatten, Dense, Lambda, Reshape
from keras.layers import BatchNormalization
from keras.models import Model
from keras.datasets import mnist
from keras.losses import binary_crossentropy
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard
from keras.models import load_model

def kl_reconstruction_loss(true, pred):
	# Reconstruction loss
	reconstruction_loss = binary_crossentropy(K.flatten(true), K.flatten(pred)) * img_width * img_height
	# KL divergence loss
	kl_loss = 1 + sigma - K.square(mu) - K.exp(sigma)
	kl_loss = K.sum(kl_loss, axis=-1)
	kl_loss *= -0.5
	# Total loss = 50% rec + 50% KL divergence loss
	return K.mean(reconstruction_loss + kl_loss)


#load model
vae = load_model('.\\vae_model.h5')

# define testimg
(x_train, y_train), (x_test, y_test) = mnist.load_data()
img_width, img_height = x_train.shape[1], x_train.shape[2]

# reshape
x_train = x_train.reshape(x_train.shape[0], img_height, img_width, 1)
x_test = x_test.reshape(x_test.shape[0], img_height, img_width, 1)
input_shape = (img_height, img_width, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train = x_train / 255
x_test = x_test / 255

loaded_model = vae.compile(loss=kl_reconstruction_loss, optimizer='adam')
score = vae.evaluate(x_test, y_test, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

mu, _, _ = vae.predict(x_test)
plt.figure(figsize=(8, 10))
plt.scatter(mu[:, 0], mu[:, 1], c=y_test)
plt.xlabel('z - dim 1')
plt.ylabel('z - dim 2')
plt.colorbar()
plt.show()