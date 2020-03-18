from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D 
from keras.layers import Activation, Dropout, Flatten, Dense 
from keras import layers, preprocessing, models
from keras import backend as K
import matplotlib.pyplot as plt
from keras import callbacks, optimizers
from keras.callbacks import TensorBoard
import cv2
import os
import numpy as np
from keras import applications

img_width, img_height = 128, 128
  
train_data_dir = 'C:\\Users\\tim.reicheneder\\Desktop\\Bachelorthesis\\impl_final\\classification\\pictures\\Train'
validation_data_dir = 'C:\\Users\\tim.reicheneder\\Desktop\\Bachelorthesis\\impl_final\\classification\\pictures\\Validation'
nb_train_samples = 117
nb_validation_samples = 37
epochs = 50
batch_size = 64
  
es = callbacks.EarlyStopping(   
    monitor='val_loss', 
    min_delta=0, 
    patience=0, 
    verbose=0, 
    mode='auto', 
    baseline=None, 
    restore_best_weights=False)

# ## build model
# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(32, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# model.add(Flatten()) 
# model.add(Dense(64)) 
# model.add(Activation('relu')) 
# model.add(Dropout(0.5)) 
# model.add(Dense(1, activation='sigmoid')) 

# model.summary()

# model.compile(
#     loss ='binary_crossentropy', 
#     optimizer ='adam', 
#     metrics =['accuracy'])

# ## load img data
# train_datagen = ImageDataGenerator( 
#     rescale = 1. / 255, 
#     shear_range = 0.2, 
#     zoom_range = 0.2, 
#     horizontal_flip = True,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     fill_mode="nearest") 
  
# test_datagen = ImageDataGenerator(rescale = 1. / 255) 
  
# train_generator = train_datagen.flow_from_directory(
#     train_data_dir, 
#     target_size =(img_width, img_height), 
#     batch_size = batch_size, 
#     class_mode ='binary') 
  
# validation_generator = test_datagen.flow_from_directory( 
#     validation_data_dir, 
#     target_size =(img_width, img_height), 
#     batch_size = batch_size, 
#     class_mode ='binary')

# ## train
# model.fit_generator(
#     train_generator, 
#     steps_per_epoch = 10, 
#     epochs = 10, 
#     validation_data = validation_generator, 
#     validation_steps = 10,
#     callbacks = [es, TensorBoard(log_dir='C:\\Users\\tim.reicheneder\\Desktop\\Bachelorthesis\\impl_final\\classification\\tmp\\classification_17.11', histogram_freq=1)]) 
    
# model.save_weights('C:\\Users\\tim.reicheneder\\Desktop\\Bachelorthesis\\impl_final\\classification\\model_weighhts_saved.h5')
# model.save('C:\\Users\\tim.reicheneder\\Desktop\\Bachelorthesis\\impl_final\\classification\\model_saved.h5')

for filename in os.listdir("C:\\Users\\tim.reicheneder\\Desktop\\Bachelorthesis\\impl_final\\classification\\pictures\\Test"):
    src = "C:\\Users\\tim.reicheneder\\Desktop\\Bachelorthesis\\impl_final\\classification\\pictures\\Test\\" + filename

    img = cv2.imread(src)
    img = cv2.resize(img, (128, 128))
    plt.imshow(img)
    plt.show()
    img = np.expand_dims(img, axis=0)

    model = models.load_model('C:\\Users\\tim.reicheneder\\Desktop\\Bachelorthesis\\impl_final\\classification\\model_saved.h5', compile = False)

    classifier = model.predict(img)

    print(filename)
    print(classifier)
