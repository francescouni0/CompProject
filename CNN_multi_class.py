import reading
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Dense, Flatten, InputLayer, Activation, Dropout
from tensorflow.keras.layers import Conv2D, Conv3D, AveragePooling2D, AveragePooling3D, MaxPooling2D, MaxPooling3D,GlobalAvgPool2D
from tensorflow.keras.optimizers import Adam, SGD
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import roc_curve, auc
import os
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow import keras





class CNN_multi(keras.Sequential):

    


    def __init__(self,images,labels):
        super().__init__()
        
        X_train, x_test, Y_train, y_test = train_test_split(images[:,:,:], labels, test_size=0.2, random_state=10)
        x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.25, random_state=20)
        
        epochs= 1000
        batch_size = 10
        
        
        reduce_on_plateau = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.1,
        patience=20,
        verbose=0,
        mode="auto",
        min_delta=0.0001,
        cooldown=0,
        min_lr=0)
    
        early_stopping = EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=100,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=False,
        start_from_epoch=10)
        
        shape=(110, 110, 3)
 
       

    def build_model(self):
        
        shape=(110, 110, 3)
        model = Sequential([
      
      Conv2D(8, (3,3), padding='valid', input_shape=shape, kernel_regularizer='l1'),
      BatchNormalization(),
      Activation('relu'),
      
      MaxPooling2D((2,2), strides=3),
      Dropout(0.3),
      
      Conv2D(16, (3,3), padding='valid', input_shape=shape, kernel_regularizer='l1'),
      BatchNormalization(),
      Activation('relu'),
      
      MaxPooling2D((3,3), strides=3),
      #Dropout(0.1),
      
      
      Flatten(),

      Dense(116, activation='relu'),
      #Dropout(0.1),
      Dense(1, activation='sigmoid')
     ])
        model.summary()
        model.compile(optimizer=SGD(learning_rate=0.01),loss=losses.Hinge(),metrics=['accuracy'])
        
        


    def train(self, x_train, y_train):
        
        model = build_model(self)
        
        model.fit(x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        steps_per_epoch = round(len(x_train)/batch_size),
        verbose=1,
        validation_data=(x_val, y_val),
        validation_steps=round(len(x_val)/batch_size),
        callbacks=[reduce_on_plateau, early_stopping])

    def predict(self, x_test):
        model.predict(x_test)

    def validation(self, x_val, y_val):
        model.evaluate(x_test,y_test, verbose=2)