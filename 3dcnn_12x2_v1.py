#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 13:31:12 2018

@author: fgimbert
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random


import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import keras
from keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout, Input, BatchNormalization
from keras.models import Model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, Activation
from keras.models import Model

from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras import backend
seed = 123



def rmsle(h, y): 
    """
    Compute the Root Mean Squared Log Error for hypthesis h and targets y

    Args:
        h - numpy array containing predictions with shape (n_samples, n_targets)
        y - numpy array containing targets with shape (n_samples, n_targets)
    """
    return np.sqrt(np.square(np.log(h + 1) - np.log(y + 1)).mean())



def soft_acc(y_true, y_pred):
    return backend.mean(backend.equal(backend.round(y_true), backend.round(y_pred)))

def get_callbacks(filepath, patience=2):
    es = EarlyStopping('val_loss', patience=patience, mode="min")
    msave = ModelCheckpoint(filepath, save_best_only=True)
    return [es, msave]
    
file_path = "./model_weights_12x2_v1.hdf5"
callbacks = get_callbacks(filepath=file_path, patience=100)


# load the data
# Load the dictionary back from the pickle file.
import pickle 
atom_train = pickle.load(open( "../input_3d_gauss/atom_train_gauss_p1.pkl", "rb" ) )
atom_train_2 = pickle.load(open( "../input_3d_gauss/atom_train_gauss_p2.pkl", "rb" ) )
atom_train_3 = pickle.load(open( "../input_3d_gauss/atom_train_gauss_p3.pkl", "rb" ) )
atom_train_4 = pickle.load(open( "../input_3d_gauss/atom_train_gauss_p4.pkl", "rb" ) )
atom_train_5 = pickle.load(open( "../input_3d_gauss/atom_train_gauss_p5.pkl", "rb" ) )

atom_train =np.concatenate((atom_train,atom_train_2), axis=0)
atom_train =np.concatenate((atom_train,atom_train_3), axis=0)
atom_train =np.concatenate((atom_train,atom_train_4), axis=0)
atom_train =np.concatenate((atom_train,atom_train_5), axis=0)



# input image dimensions
n=24
img_x, img_y , img_z = n, n, n

atom_train=atom_train.reshape(-1,img_x, img_y , img_z,4)
print(atom_train.shape)

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

#print(test.head())

t1 = 'formation_energy_ev_natom'
t2 = 'bandgap_energy_ev'

feature_columns = [ 'number_of_total_atoms', 'percent_atom_al', 'percent_atom_ga', 'percent_atom_in', 'lattice_vector_1_ang', 'lattice_vector_2_ang', 'lattice_vector_3_ang', 'lattice_angle_alpha_degree', 'lattice_angle_beta_degree', 'lattice_angle_gamma_degree']

nb_features=len(feature_columns)
print('nb_features :', nb_features)

#print(train.head())

all_columns = [t1, t2, *feature_columns]

all = pd.concat([train[feature_columns], test[feature_columns]])

X_train_split, X_val, y_train_split, y_val = train_test_split(atom_train, train[[t1, t2]], 
                                                    train_size=0.8, random_state=87)
           

X_train = np.array(atom_train)
y_train = np.array(train[[t1, t2]])

print(X_train.shape)
print(y_train.shape)


# Number of Classes and Epochs of Training
nb_classes = 2 # cube, cone or sphere
nb_epoch = 50
batch_size = 20


# Number of Convolutional Filters to use
nb_filters = 32

# Convolution Kernel Size
kernel_size = [4,4,4]

# (25 rows, 25 cols, 25 of depth,4 channels)
input_shape = (img_x, img_y , img_z ,4)

# Init
model = Sequential()

# 3D Convolution layer

model.add(Conv3D(64, (5, 5, 5), input_shape=(img_x, img_y , img_z ,4)))
#model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv3D(64, (5, 5, 5)))
#model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(Dropout(0.2))

model.add(Conv3D(32, (3, 3, 3)))
#model.add(BatchNormalization())
model.add(Activation('relu'))


model.add(MaxPooling3D(pool_size=(2, 2, 2)))

model.add(Conv3D(32, (3, 3, 3)))
#model.add(BatchNormalization())
model.add(Activation('relu'))

#model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(Dropout(0.2))

# Fully Connected layer
model.add(Flatten())
model.add(Dense(128,kernel_initializer='normal'))
#model.add(BatchNormalization())
model.add(Activation('relu'))

#model.add(Dropout(0.2))
#model.add(Dense(128,kernel_initializer='normal',activation='relu'))
# Output Layer
model.add(Dense(nb_classes,kernel_initializer='normal',activation='relu'))
model.summary()

# Compile
model.compile(loss='mse', optimizer='adam', metrics=[soft_acc])

# Fit network
history=model.fit(X_train, y_train, batch_size=50, epochs=nb_epoch,validation_split=0.2,
         verbose=1,callbacks=callbacks)

# Plot the loss and accuracy curves for training and validation 
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['soft_acc'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_soft_acc'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)
plt.show()
          
#model.load_weights(filepath=file_path)
#validation_scores = model.evaluate(model.validation_data[0],model.validation_data[1], verbose=0)

#print('Validation loss:', validation_scores[0])
#print('Validation accuracy:', validation_scores[1])


## Predict the values from the test dataset
#
#import pickle 



atom_test = pickle.load(open( "../input_3d_gauss/atom_test_gauss_p1.pkl", "rb" ) )
atom_test_2 = pickle.load(open( "../input_3d_gauss/atom_test_gauss_p2.pkl", "rb" ) )
atom_test_3 = pickle.load(open( "../input_3d_gauss/atom_test_gauss_p3.pkl", "rb" ) )
atom_test_4 = pickle.load(open( "../input_3d_gauss/atom_test_gauss_p4.pkl", "rb" ) )
atom_test_5 = pickle.load(open( "../input_3d_gauss/atom_test_gauss_p5.pkl", "rb" ) )

atom_test =np.concatenate((atom_test,atom_test_2), axis=0)
atom_test =np.concatenate((atom_test,atom_test_3), axis=0)
atom_test =np.concatenate((atom_test,atom_test_4), axis=0)
atom_test =np.concatenate((atom_test,atom_test_5), axis=0)


X_test=atom_test.reshape(-1,img_x, img_y , img_z,4)
print(atom_test.shape)
#
#


Y_pred = model.predict(X_test)

X_train_split, X_val, y_train_split, y_val = train_test_split(atom_train, train[[t1, t2]], 
                                                    train_size=0.8, random_state=87)

Y_val_pred = model.predict(X_val)

rmsle_result = rmsle(Y_val_pred,y_val)


print(rmsle_result)
#
## sample submission
sample = pd.read_csv('../input/sample_submission.csv')
#sample.head()
#
##pred_y = np.expm1(Y_pred)
pred_y = Y_pred
#
pred_y[pred_y[:, 0] < 0, 0] = 0
pred_y[pred_y[:, 1] < 0, 1] = 0

subm = pd.DataFrame()
subm['id'] = sample['id']
subm[t1] = pred_y[:, 0]
subm[t2] = pred_y[:, 1]
subm.head()
subm.to_csv("subm_cnn_12x2_v1.csv", index=False)
