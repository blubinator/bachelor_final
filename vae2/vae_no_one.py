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

def sample_z(args):
	mu, sigma = args
	batch     = K.shape(mu)[0]
	dim       = K.int_shape(mu)[1]
	eps       = K.random_normal(shape=(batch, dim))
	return mu + K.exp(sigma / 2) * eps

# Define loss
def kl_reconstruction_loss(true, pred):
	# Reconstruction loss
	reconstruction_loss = binary_crossentropy(K.flatten(true), K.flatten(pred)) * img_width * img_height
	# KL divergence loss
	kl_loss = 1 + sigma - K.square(mu) - K.exp(sigma)
	kl_loss = K.sum(kl_loss, axis=-1)
	kl_loss *= -0.5
	# Total loss = 50% rec + 50% KL divergence loss
	return K.mean(reconstruction_loss + kl_loss)

# visualize
def viz_latent_space(encoder, data):
	input_data, target_data = data
	mu, _, _ = encoder.predict(input_data)
	plt.figure(figsize=(8, 10))
	plt.scatter(mu[:, 0], mu[:, 1], c=target_data)
	plt.xlabel('z - dim 1')
	plt.ylabel('z - dim 2')
	plt.colorbar()
	plt.show()

def viz_decoded(encoder, decoder, data):
	num_samples = 15
	figure = np.zeros((img_width * num_samples, img_height * num_samples, num_channels))
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

def test_model():
	#load model
	vaetest = load_model('C:\\Users\\tim.reicheneder\\Desktop\\Bachelorthesis\\impl_final\\vae2\\vae_model.h5', compile=False)
	vaetest.compile(optimizer='adam', loss=kl_reconstruction_loss)
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

	first_image = x_train[3]
	first_image = np.array(first_image, dtype='float')
	pixels = first_image.reshape((28, 28))
	plt.imshow(pixels, cmap='gray')
	plt.show()


	pred = vaetest.predict(first_image.reshape(1, 28, 28, 1))

	plt.figure(figsize=(8, 10))
	plt.scatter(pred[:, 0], pred[:, 1])
	plt.xlabel('z - dim 1')
	plt.ylabel('z - dim 2')
	plt.colorbar()
	plt.show()

# load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# filter out images of '1'
train_filter = np.where(y_train != 1)
test_filter = np.where(y_test != 1)

x_train = x_train[train_filter]
x_test = x_test[test_filter]


# config
img_width, img_height = x_train.shape[1], x_train.shape[2]
batch_size = 128
epochs = 100
validation_split = 0.2
verbosity = 1
latent_dim = 2
num_channels = 1

# reshape
x_train = x_train.reshape(x_train.shape[0], img_height, img_width, num_channels)
x_test = x_test.reshape(x_test.shape[0], img_height, img_width, num_channels)
input_shape = (img_height, img_width, num_channels)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train = x_train / 255
x_test = x_test / 255

# encoder
i = Input(shape=input_shape, name='encoder_input')
en = Conv2D(filters=8, kernel_size=3, strides=2, padding='same', activation='relu')(i)
en = BatchNormalization()(en)
en = Conv2D(filters=16, kernel_size=3, strides=2, padding='same', activation='relu')(en)
en_n = BatchNormalization()(en)
en = Flatten()(en_n)
en = Dense(20, activation='relu')(en)
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
de = Conv2DTranspose(filters=16, kernel_size=3, strides=2, padding='same', activation='relu')(de)
de = BatchNormalization()(de)
de = Conv2DTranspose(filters=8, kernel_size=3, strides=2, padding='same',  activation='relu')(de)
de = BatchNormalization()(de)
de = Conv2DTranspose(filters=num_channels, kernel_size=3, activation='sigmoid', padding='same', name='decoder_output')(de)

# Instantiate decoder
decoder = Model(de_i, de, name='decoder')
decoder.summary()

# VAE
vae_outputs = decoder(encoder(i)[2])
vae = Model(i, vae_outputs, name='vae')
vae.summary()

# Compile VAE
vae.compile(optimizer='adam', loss=kl_reconstruction_loss)

# Train autoencoder
vae.fit(x_train, x_train, 
		epochs = epochs, 
		batch_size = batch_size, 
		validation_split = validation_split,
		callbacks=[TensorBoard(log_dir='.\\tmp\\autoencoder_without_1', histogram_freq=1)])

vae.save('vae_model_without_1.h5')

# Plot results
data = (x_test, y_test)
viz_latent_space(encoder, data)
viz_decoded(encoder, decoder, data)

# call test with 1
test_model()
