import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
import matplotlib.pyplot as plt
from keras import backend as K
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# Define the model
model = Sequential()

#1st convolution layer
model.add(Conv2D(16, (3, 3), padding='same', input_shape=(224,224,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), padding='same'))

#2nd convolution layer
model.add(Conv2D(2,(3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
#-------------------------
#3rd convolution layer
model.add(Conv2D(2,(3, 3), padding='same'))
model.add(Activation('relu'))
model.add(UpSampling2D((2, 2)))

#4rd convolution layer
model.add(Conv2D(16,(3, 3), padding='same'))
model.add(Activation('relu'))
model.add(UpSampling2D((2, 2)))

#-------------------------

model.add(Conv2D(3,(3, 3), padding='same'))
model.add(Activation('sigmoid'))

model.summary()

# Compile the model
model.compile(optimizer='adadelta', loss='binary_crossentropy')

# Generate data from the images in a folder
batch_size = 8
train_datagen = ImageDataGenerator(rescale=1./255, data_format='channels_last')
train_generator = train_datagen.flow_from_directory(
    'C:\\Users\\tim.reicheneder\\Desktop\\Bachelorthesis\\impl_final\\pictures\\',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='input'
    )
test_datagen = ImageDataGenerator(rescale=1./255, data_format='channels_last')
validation_generator = test_datagen.flow_from_directory(
    'C:\\Users\\tim.reicheneder\\Desktop\\Bachelorthesis\\impl_final\\original_pictures\\',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='input'
    )
    
# Train the model
model.fit_generator(
        train_generator,
        steps_per_epoch=1000 // batch_size,
        epochs=20,
        validation_data=validation_generator,
        validation_steps=1000 // batch_size)
        

# Test the model
data_list = []
batch_index = 0
while batch_index <= train_generator.batch_index:
    data = train_generator.next()
    data_list.append(data[0])
    batch_index = batch_index + 1
data_list[0].shape

predicted = model.predict(data_list[0])
plt.imshow(data_list[0][0])
plt.imshow(predicted[0])