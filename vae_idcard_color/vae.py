#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras
from keras.layers import Conv2D, Conv2DTranspose, Input, Flatten, Dense, Lambda, Reshape
from keras.layers import BatchNormalization
from keras.models import Model
from keras.datasets import mnist
from keras.losses import binary_crossentropy
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard
from keras.models import load_model
import cv2
from keras.utils import plot_model
import pandas as pd
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import train_test_split
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# In[2]:


def sample_z(args):
	mu, sigma = args
	batch = K.shape(mu)[0]
	dim = K.int_shape(mu)[1]
	eps = K.random_normal(shape=(batch, dim))
	return mu + K.exp(sigma / 2) * eps


# In[3]:


def kl_reconstruction_loss(true, pred):
	# Reconstruction loss
	reconstruction_loss = binary_crossentropy(
	    K.flatten(true), K.flatten(pred)) * img_width * img_height
	# KL divergence loss
	kl_loss = 1 + sigma - K.square(mu) - K.exp(sigma)
	kl_loss = K.sum(kl_loss, axis=-1)
	kl_loss *= -0.5
	# Total loss = 50% rec + 50% KL divergence loss
	return K.mean(reconstruction_loss + kl_loss)


# In[4]:


def model_mse(x, vae):
    pred = vae.predict(x, batch_size = batch_size)
    sq = np.square(pred)
    mean = np.mean(sq, (1,2,3))
    return mean


# In[5]:


def histrogram(x_train, x_test, anomaly_x_test):
    vae = load_model(
    'C:\\Users\\tim.reicheneder\\Desktop\\Bachelorthesis\\impl_final\\vae_with_idcard\\vae.h5', compile = False)
    vae.compile(optimizer = 'adam', loss = kl_reconstruction_loss)

    # calc error
    fig, ax1 = plt.subplots(1,1, figsize = (8,8))
    ax1.hist(model_mse(x_train, vae), alpha = 1.0, label = 'Training data', normed = True)
    ax1.hist(model_mse(x_test, vae), alpha = 0.5, label = 'Testing data', normed = True)
    ax1.hist(model_mse(anomaly_x_test, vae), alpha = 0.5, label = 'Anomaly data', normed = True)
    ax1.legend()
    ax1.set_xlabel('Reconstruction Error')
    plt.show()


# In[6]:


def reconstruction_comparison(x_test, anomaly_x_test):
    # load model
    vae = load_model(
        'C:\\Users\\tim.reicheneder\\Desktop\\Bachelorthesis\\impl_final\\vae_with_idcard\\vae.h5', compile = False)
    vae.compile(optimizer = 'adam', loss = kl_reconstruction_loss)

    # reconstruction
    fig, m_axs = plt.subplots(5,4, figsize=(20, 10))
    [c_ax.axis('off') for c_ax in m_axs.ravel()]
    for i, (axa_in, axa_vae, axt_in, axt_vae) in enumerate(m_axs):
        axa_in.imshow(np.squeeze(anomaly_x_test[i]))
        axa_in.set_title('Anomaly In')
        axa_vae.imshow(vae.predict(anomaly_x_test[i:i+1])[0,:,:,0])
        axa_vae.set_title('Anomaly/Reconstructed')
        axt_in.imshow(np.squeeze(x_test[i]))
        axt_in.set_title('Test In')
        axt_vae.imshow(vae.predict(x_test[i:i+1])[0,:,:,0])
        axt_vae.set_title('Test Reconstructed')

    plt.show()


# In[7]:


def viz_latent_space(encoder, data):
	input_data, target_data = data
	mu, _, _ = encoder.predict(input_data, batch_size = batch_size)
	plt.figure(figsize=(10, 10))
	plt.scatter(mu[:, 0], mu[:, 1], c=target_data)
	plt.xlabel('z - dim 1')
	plt.ylabel('z - dim 2')
	plt.colorbar()
	plt.show()


# In[8]:


def make_outliers_colored(target_data):
    cols=[]
    for val in target_data:
        if val == 1:
            cols.append('red')
        elif val == 7:
            cols.append('blue')
        else:
            cols.append('green')

    return cols


# In[9]:


def viz_decoded(encoder, decoder, data):
	num_samples = 15
	figure = np.zeros(
	    (img_width * num_samples, img_height * num_samples, num_channels))
	grid_x = np.linspace(-4, 4, num_samples)
	grid_y = np.linspace(-4, 4, num_samples)[::-1]
	for i, yi in enumerate(grid_y):
		for j, xi in enumerate(grid_x):
			z_sample = np.array([[xi, yi]])
			x_decoded = decoder.predict(z_sample)
			digit = x_decoded[0].reshape(img_width, img_height, num_channels)
			figure[i * img_width: (i + 1) * img_width,
					j * img_height: (j + 1) * img_height] = digit
	plt.figure(figsize=(10, 10))
	start_range = img_width // 2
	end_range = num_samples * img_width + start_range + 1
	pixel_range = np.arange(start_range, end_range, img_width)
	sample_range_x = np.round(grid_x, 1)
	sample_range_y = np.round(grid_y, 1)
	plt.xticks(pixel_range, sample_range_x)
	plt.yticks(pixel_range, sample_range_y)
	plt.xlabel('z - dim 1')
	plt.ylabel('z - dim 2')
	# matplotlib.pyplot.imshow() needs a 2D array, or a 3D array with the third dimension being of shape 3 or 4!
	# So reshape if necessary
	fig_shape = np.shape(figure)
	if fig_shape[2] == 1:
		figure = figure.reshape((fig_shape[0], fig_shape[1]))
	# Show image
	plt.imshow(figure)
	plt.show()


# In[10]:


def getAndPrepareImages(path) :
    print('data prep begins')
    img_array = []
    for filename in os.listdir(path):
        temp = cv2.imread(path + filename)
        temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
        temp = cv2.resize(temp, (800, 600))
        img_array.append(temp)

    img_array = np.array(img_array, dtype="uint8")

    return img_array


# In[11]:


# config // hyperparameter
batch_size = 16
epochs = 50
validation_split = 0.2
verbosity = 1
latent_dim = 2

img_width, img_height = 800, 600
num_channels = 1
input_shape = (img_height, img_width, num_channels)


# In[12]:


def prepare_data():
    train_datagen = ImageDataGenerator( 
    rescale = 1. / 255,  
    brightness_range=[0.7, 1.3],
    validation_split=0.2)
    
    test_datagen = ImageDataGenerator(rescale = 1. / 255)
    
    train_generator = train_datagen.flow_from_directory(
        "C:\\Users\\tim.reicheneder\\Desktop\\Bachelorthesis\\impl_final\\pictures_idcard\\preprocessed\\",
        target_size=(600, 800),
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='input',
        subset='training')
    
    validation_generator = train_datagen.flow_from_directory(
        "C:\\Users\\tim.reicheneder\\Desktop\\Bachelorthesis\\impl_final\\pictures_idcard\\preprocessed\\",
        target_size=(600, 800),
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='input',
        subset='validation')
    
    return train_generator, validation_generator


# In[13]:


def define_model():
    i = Input(shape=input_shape, name='encoder_input')
    
    en = Conv2D(filters=num_channels, kernel_size=2, padding='same', activation='relu')(i)
    en = BatchNormalization()(en)
    
    en = Conv2D(filters=32, kernel_size=3, strides=2, padding='same', activation='relu')(en)
    en = BatchNormalization()(en)
    
    en = Conv2D(filters=64, kernel_size=3, strides=2, padding='same', activation='relu')(en)
    en = BatchNormalization()(en)

    en = Conv2D(filters=128, kernel_size=3, strides=2, padding='same', activation='relu')(en)
    en_n = BatchNormalization()(en)
    
    en = Flatten()(en_n)
    
    en = Dense(128, activation='relu')(en)
    en = BatchNormalization()(en)
    
    mu = Dense(latent_dim, name='latent_mu')(en)
    sigma = Dense(latent_dim, name='latent_sigma')(en)

    # get conv shape
    conv_shape = K.int_shape(en_n)

    # Use reparameterization trick to ensure correct gradient
    z = Lambda(sample_z, output_shape=(latent_dim, ), name='z')([mu, sigma])

    # Instantiate encoder
    encoder = Model(i, [mu, sigma, z], name='encoder')
    encoder.summary()

    # decoder
    de_i = Input(shape=(latent_dim, ), name='decoder_input')

    de = Dense(conv_shape[1] * conv_shape[2] * conv_shape[3], activation='relu')(de_i)
    de = BatchNormalization()(de)
    
    de = Reshape((conv_shape[1], conv_shape[2], conv_shape[3]))(de)
    
    de = Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding='same', activation='relu')(de)
    de = BatchNormalization()(de)
    
    de = Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation='relu')(de)
    de = BatchNormalization()(de)
    
    de = Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', activation='relu')(de)
    de = BatchNormalization()(de)
    
    de = Conv2DTranspose(filters=num_channels, kernel_size=3, activation='sigmoid', padding='same', name='decoder_output')(de)

    # Instantiate decoder
    decoder = Model(de_i, de, name='decoder')
    decoder.summary()

    # VAE
    vae_outputs = decoder(encoder(i)[2])
    vae = Model(i, vae_outputs, name='vae')
    vae.summary()
    
    return encoder, decoder, vae, sigma, mu


# In[14]:


from datetime import datetime

def comp_and_train(encoder, decoder, vae):
    vae.compile(optimizer='adam', loss=kl_reconstruction_loss)

    # Train autoencoder
    logdir = 'C:\\Users\\tim.reicheneder\\Desktop\\Bachelorthesis\\impl_final\\vae_idcard_color\\tmp\\vae_' + datetime.now().strftime("%H_%M")
    
    print('train_samples = ', train_generator.samples)
    
    vae.fit_generator(
        generator = train_generator,
        steps_per_epoch = train_generator.samples/batch_size,
        validation_data = validation_generator,
        validation_steps = validation_generator.samples/batch_size,
        epochs = epochs,
        callbacks = [TensorBoard(log_dir= logdir, histogram_freq=1)]) 
    
    vae.save('C:\\Users\\tim.reicheneder\\Desktop\\Bachelorthesis\\impl_final\\vae_idcard_color\\vae.h5')

    encoder.save('C:\\Users\\tim.reicheneder\\Desktop\\Bachelorthesis\\impl_final\\vae_idcard_color\\encoder.h5')

    decoder.save('C:\\Users\\tim.reicheneder\\Desktop\\Bachelorthesis\\impl_final\\vae_idcard_color\\decoder.h5')


# In[15]:


encoder, decoder, vae, sigma, mu = define_model()


# In[16]:


train_generator, validation_generator = prepare_data()


# In[ ]:


comp_and_train(encoder, decoder, vae)


# In[ ]:




