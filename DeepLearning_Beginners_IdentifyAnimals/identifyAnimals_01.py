# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 11:40:31 2018

@author: ThapaRak
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from keras.utils import to_categorical


train_img_dir = r'C:\Users\ThapaRak\Documents\MyCodes\ML_Exercises\DeepLearning_Beginners_IdentifyAnimals\dataset\temp_dir\temp_train_dir'
test_img_dir =r'C:\Users\ThapaRak\Documents\MyCodes\ML_Exercises\DeepLearning_Beginners_IdentifyAnimals\dataset\temp_dir\temp_test_dir'

train_Y_file = r'C:\Users\ThapaRak\Documents\MyCodes\ML_Exercises\DeepLearning_Beginners_IdentifyAnimals\dataset\DL#+Beginner\DL# Beginner\meta-data\meta-data\train.csv'
test_Y_file = r'C:\Users\ThapaRak\Documents\MyCodes\ML_Exercises\DeepLearning_Beginners_IdentifyAnimals\dataset\DL#+Beginner\DL# Beginner\meta-data\meta-data\test.csv' 

train_Y_data = pd.read_csv(train_Y_file)
test_Y_data = pd.read_csv(test_Y_file)

train_X_data = []
test_X_data = []

for img in train_Y_data.Image_id:
    img = os.path.join(train_img_dir, img)
    img = cv2.imread(img)
    train_X_data.append(img)
    
for img in test_Y_data.Image_id:
    img = os.path.join(test_img_dir, img)
    img = cv2.imread(img)
    test_X_data.append(img)
        
# print the shape
# print('Training data shape:', train_X_data.shape, train_Y_data.shape)
# print('Testing data shape:', test_X_data.shape, test_Y_data.shape)

# Find the unique numbers from the class labels
classes = np.unique(train_Y_data.Animal)
nClasses = len(classes)

print('Total number of outputs: ' , nClasses)
print('Output classes:', classes)

# Shape of training and test datasets# Shape  
print ('Training dataset consists of {} images with {} attributes'.format(train_Y_data.shape[0], train_Y_data.shape[1]-1))
# Shape of training and test datasets
print ('Testing dataset consists of {} images.'.format(test_Y_data.shape[0]))

# Data Preprocessing
# Convert each 128x128x3 image of the train and test set into a matric of 128 x 128 x 3 which is fed into the network
train_X = np.array(train_X_data)
test_X = np.array(test_X_data)
train_X = train_X.reshape(-1, 128,128, 3)
test_X = test_X.reshape(-1, 128,128, 3)

print('Data Shape:', train_X.shape, test_X.shape)

train_X = train_X.astype('float32')
test_X = test_X.astype('float32')
train_X = train_X / 255.
test_X = test_X / 255.

#train_X = np.array(train_X_data, np.float32) / 255.
#test_X = np.array(test_X_data, np.float32) / 255.

# Garbage collector
import gc
del train_X_data, test_X_data
gc.collect()

"""
train_X_mean_img = train_X.mean(axis=0)
train_X_std_dev = train_X.std(axis = 0)
train_X_norm = (train_X - train_X_mean_img)/ train_X_std_dev
train_X_norm.shape

test_X_mean_img = test_X.mean(axis=0)
test_X_std_dev = test_X.std(axis = 0)
test_X_norm = (test_X - test_X_mean_img)/ test_X_std_dev
test_X_norm.shape

del train_X, train_X_mean_img,train_X_std_dev, test_X, test_X_mean_img, test_X_std_dev
gc.collect()
"""

animals_set = set(train_Y_data.Animal)
train_Y_dict = {}
i=0
for animal in animals_set:
    train_Y_dict[animal] = i
    i += 1


# Add the column to the train_Y_Data. Initialize it to np.nan
train_Y_data['AnimalID'] = train_Y_data['Animal'].map(train_Y_dict)
#train_Y_data['AnimalID'] = np.nan
#train_Y_data['AnimalID'] = train_Y_data.apply(lambda x: train_Y_dict[x['Animal']], axis=1)
    
# Change the labels from categorical to one-hot encoding
train_Y_one_hot = to_categorical(train_Y_data.AnimalID)
#test_Y_one_hot = to_categorical(test_Y_data.AnimalID)


# Display the change for category label using one-hot encoding
print('Original label:',train_Y_data.Animal[0], ' Animal:', train_Y_data.AnimalID[0])
print('After conversion to one-hot:', train_Y_one_hot[0])

from sklearn.model_selection import train_test_split
train_X,valid_X,train_label,valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=13)
train_X.shape,valid_X.shape,train_label.shape,valid_label.shape

#del train_X_norm
#gc.collect()

import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

batch_size = 64
epochs = 10
num_classes = 30

animal_model = Sequential()
animal_model.add(BatchNormalization(input_shape=train_X.shape[1:]))
animal_model.add(Conv2D(32, kernel_size=(3, 3), activation= 'relu'))

animal_model.add(Conv2D(64, kernel_size=(3, 3), activation= 'relu', padding='same'))
animal_model.add(Conv2D(64, kernel_size=(3, 3), activation= 'relu'))
animal_model.add(MaxPooling2D(pool_size=(2,2)))
animal_model.add(Dropout(0.25))

animal_model.add(Conv2D(128, kernel_size=(3, 3), activation= 'relu', padding='same'))
animal_model.add(Conv2D(128, kernel_size=(3, 3), activation= 'relu'))
animal_model.add(MaxPooling2D(pool_size=(2,2)))
animal_model.add(Dropout(0.25))

animal_model.add(Flatten())
animal_model.add(Dropout(0.5))
animal_model.add(Dense(num_classes, activation='sigmoid'))

animal_model.summary()

# Compile the model and fit on the training and validation on validation data
animal_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

early_stops = keras.callbacks.EarlyStopping(patience=3, monitor='val_acc')
checkpointer = keras.callbacks.ModelCheckpoint(filepath='weights.best.eda.hdf5', verbose=1, save_best_only=True)

animal_train_dropout = animal_model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label), callbacks=[checkpointer])
# Saving the Model Data. For later use.
animal_model.save("animal_model_dropout.h5py")

# Model Evaluation on the Test set
#test_eval = animal_model.evaluate(test_X, test_Y_one_hot, verbose=1)
#print('Test loss:', test_eval[0])
#print('Test accuracy:', test_eval[1])

# PLotting the graph for Test loss and Validation Loss
accuracy = animal_train_dropout.history['acc']
val_accuracy = animal_train_dropout.history['val_acc']
loss = animal_train_dropout.history['loss']
val_loss = animal_train_dropout.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

# Predict Labels
predicted_classes = animal_model.predict(test_X)

# Convert the floating point to the integer.
# np.argmax() selects the index number which has highest value in the row
predicted_classes = np.argmax(np.round(predicted_classes), axis=1)
predicted_classes.shape


# Now predict on the Test File
animal_model.load_weights('animal_model_dropout.h5py')

print('End')