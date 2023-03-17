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
        epochs= 1000
        batch_size = 10
        
        X_train, x_test, Y_train, y_test = train_test_split(images[:,:,:], labels, test_size=0.2, random_state=10)
        x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.25, random_state=20)
        
        
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
        X_train, x_test, Y_train, y_test = train_test_split(images[:,:,:], labels, test_size=0.2, random_state=10)
        x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.25, random_state=20)
        
        
        model.predict(x_test)

    def validation(self, x_val, y_val):
        X_train, x_test, Y_train, y_test = train_test_split(images[:,:,:], labels, test_size=0.2, random_state=10)
        x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.25, random_state=20)        
        
        
        model.evaluate(x_test,y_test, verbose=2)











        """Prova Prova
        """
        
paths_FA= reading.data_path("Diffusion_parameters_maps-20230215T134959Z-001","corrected_FA_image")
paths_MD= reading.data_path("Diffusion_parameters_maps-20230215T134959Z-001","corrected_MD_image")
paths_AD= reading.data_path("Diffusion_parameters_maps-20230215T134959Z-001","corrected_AD_image")
paths_RD= reading.data_path("Diffusion_parameters_maps-20230215T134959Z-001","corrected_RD_image")

#paths_FA.sort(key=lambda x: int(os.path.basename(x).split('_')[3]))
#paths_MD.sort(key=lambda x: int(os.path.basename(x).split('_')[3]))
#paths_AD.sort(key=lambda x: int(os.path.basename(x).split('_')[3]))
#paths_RD.sort(key=lambda x: int(os.path.basename(x).split('_')[3]))

dataset=pd.DataFrame(pd.read_csv('ADNI_dataset_diffusion.csv'))
dataset.sort_values(by=["Subject"],inplace=True,ignore_index=True)
dataset["Path FA"]=paths_FA
dataset["Path MD"]=paths_MD
dataset["Path AD"]=paths_AD
dataset["Path RD"]=paths_RD
pd.set_option("max_colwidth",None)       


images_list = []
k_slice = 45

for i, datapath in enumerate(dataset["Path FA"]):
    image_FA = np.asarray(nib.load(dataset["Path FA"][i]).get_fdata())
    image_MD = np.asarray(nib.load(dataset["Path MD"][i]).get_fdata())
    image_AD = np.asarray(nib.load(dataset["Path AD"][i]).get_fdata())
    image = np.stack((image_FA[k_slice], image_MD[k_slice], image_AD[k_slice]), axis=-1)
    images_list.append(image)
    
images = np.array(images_list, dtype='float64')
labels = np.array(dataset["Group"], dtype='int64')

print(np.shape(images))



augmentation_rot = Sequential([layers.RandomRotation((-0.5,0.5))])
augmentation_zoom = Sequential([layers.RandomZoom(0.5)])
augmentation_crop = Sequential([layers.RandomCrop(110, 110, seed=3)])
augmentation_cont = Sequential([layers.RandomContrast(1,seed=5)])
augmentation_zoom2 = Sequential([layers.RandomZoom(0.6)])
augmentation_zoom3 = Sequential([layers.RandomZoom(0.7)])
augmentation_zoom4 = Sequential([layers.RandomZoom(0.65)])
augmentation_cont2 = Sequential([layers.RandomContrast(0.8,seed=8)])





images_rotated = augmentation_rot(images)
images_zoomed = augmentation_zoom(images)
images_croped = augmentation_crop(images)
images_contr= augmentation_cont(images)
images_zoomed2 = augmentation_zoom2(images)
images_zoomed3=augmentation_zoom3(images)
images_zoomed4=augmentation_zoom4(images)
images_contr2= augmentation_cont2(images)



images = np.concatenate((images, images_rotated, images_zoomed, images_croped, images_contr,images_zoomed2,images_zoomed3,images_zoomed4,images_contr2), axis = 0)

labels = np.concatenate((labels, labels, labels,labels,labels,labels,labels,labels,labels))


        
from CNN_multi_class import CNN_multi

model=CNN_multi(images,labels)

model.build_model()
model.summary()
model.train()

