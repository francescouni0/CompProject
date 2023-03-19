import reading
import CNN_multi_utilities
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Dense, Flatten, InputLayer, Activation, Dropout
from tensorflow.keras.layers import Conv2D, AveragePooling2D, MaxPooling2D, GlobalAvgPool2D
from tensorflow.keras.optimizers import Adam, SGD
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import roc_curve, auc
import os
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow import keras
import tensorflow



class MyModel(tensorflow.keras.Model):
    
    def __init__(self, shape=(110, 110, 3)):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(8, (3,3), padding='valid', input_shape=shape, kernel_regularizer='l1')
        self.bn1 = BatchNormalization()
        self.act1 = Activation('relu')
        self.pool1 = MaxPooling2D((2,2), strides=3)
        self.drop1 = Dropout(0.3)
        self.conv2 = Conv2D(16, (3,3), padding='valid', input_shape=shape, kernel_regularizer='l1')
        self.bn2 = BatchNormalization()
        self.act2 = Activation('relu')
        self.pool2 = MaxPooling2D((3,3), strides=3)
        self.flatten = Flatten()
        self.fc1 = Dense(116, activation='relu')
        self.fc2 = Dense(1, activation='sigmoid')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.drop1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return self.fc2(x)
    
    def compile_and_fit(self, x_train, y_train, x_val, y_val,x_test,y_test):
        self.compile(optimizer=SGD(learning_rate=0.01), loss=losses.Hinge(), metrics=['accuracy'])
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

        epochs= 1000
        batch_size = 10
        history = self.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            steps_per_epoch = round(len(x_train)/batch_size),
                            verbose=1,
                            validation_data=(x_val, y_val),
                            validation_steps=round(len(x_val)/batch_size),
                            callbacks=[reduce_on_plateau, early_stopping])
        self.loss_plot(history), self.accuracy_roc(x_val,y_val),self.test_roc(x_test,y_test)

    def loss_plot(self, history):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(1, len(acc)+1)
        #Train and validation accuracy 
        plt.figure(figsize=(15, 15))
        plt.subplot(2, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')
        #Train and validation loss 
        plt.subplot(2, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()
        
    def accuracy_roc(self, x_val, y_val):
        _, val_acc = self.evaluate(x_val, y_val, verbose=0)
        print('Validation accuracy: %.3f' % (val_acc))
    
        preds = self.predict(x_val, verbose=1)
        # Compute Receiver operating characteristic (ROC)
        fpr, tpr, _ = roc_curve(y_val, preds)
        roc_auc = auc(fpr, tpr)
        print('AUC = %.3f'% (roc_auc))
    
        # Plot of a ROC curve
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Validation ROC')
        plt.legend(loc="lower right")
        plt.show()
        
    def test_roc(self,x_test,y_test):
        test_loss, test_acc = self.evaluate(x_test, y_test)
        print('\nTest accuracy: %.3f' % (test_acc))
        
        preds_test = self.predict(x_test, verbose=1)
        fpr, tpr, _ = roc_curve(y_test, preds_test)
        roc_auc = auc(fpr, tpr)
        print('AUC = %.3f'% (roc_auc))
        
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Test ROC')
        plt.legend(loc="lower right")
        plt.show()





        """
        Prova Prova
        """


paths_FA = reading.data_path("Diffusion_parameters_maps-20230215T134959Z-001", "corrected_FA_image")
paths_MD = reading.data_path("Diffusion_parameters_maps-20230215T134959Z-001", "corrected_MD_image")
paths_AD = reading.data_path("Diffusion_parameters_maps-20230215T134959Z-001", "corrected_AD_image")
paths_RD = reading.data_path("Diffusion_parameters_maps-20230215T134959Z-001", "corrected_RD_image")

#paths_FA.sort(key=lambda x: int(os.path.basename(x).split('_')[3]))
#paths_MD.sort(key=lambda x: int(os.path.basename(x).split('_')[3]))
#paths_AD.sort(key=lambda x: int(os.path.basename(x).split('_')[3]))
#paths_RD.sort(key=lambda x: int(os.path.basename(x).split('_')[3]))

dataset=pd.DataFrame(pd.read_csv('ADNI_dataset_diffusion.csv'))
dataset.sort_values(by=["Subject"], inplace=True, ignore_index=True)
dataset["Path FA"] = paths_FA
dataset["Path MD"] = paths_MD
dataset["Path AD"] = paths_AD
dataset["Path RD"] = paths_RD
pd.set_option("max_colwidth", None)


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

augmentation_rot = Sequential([layers.RandomRotation((-0.5, 0.5))])
augmentation_zoom = Sequential([layers.RandomZoom(0.5)])
augmentation_crop = Sequential([layers.RandomCrop(110, 110, seed=3)])
augmentation_cont = Sequential([layers.RandomContrast(1, seed=5)])
augmentation_zoom2 = Sequential([layers.RandomZoom(0.6)])
augmentation_zoom3 = Sequential([layers.RandomZoom(0.7)])
augmentation_zoom4 = Sequential([layers.RandomZoom(0.65)])
augmentation_cont2 = Sequential([layers.RandomContrast(0.8, seed=8)])

images_rotated = augmentation_rot(images)
images_zoomed = augmentation_zoom(images)
images_croped = augmentation_crop(images)
images_contr = augmentation_cont(images)
images_zoomed2 = augmentation_zoom2(images)
images_zoomed3 = augmentation_zoom3(images)
images_zoomed4 = augmentation_zoom4(images)
images_contr2 = augmentation_cont2(images)

images = np.concatenate((images, images_rotated, images_zoomed, images_croped, images_contr, images_zoomed2, images_zoomed3, images_zoomed4, images_contr2), axis=0)

labels = np.concatenate((labels, labels, labels, labels, labels, labels, labels, labels, labels))



images = np.concatenate((images, images_rotated, images_zoomed, images_croped, images_contr,images_zoomed2,images_zoomed3,images_zoomed4,images_contr2), axis = 0)

labels = np.concatenate((labels, labels, labels,labels,labels,labels,labels,labels,labels))

def split_2D():
    X_train, x_test, Y_train, y_test = train_test_split(images[:,:,:], labels, test_size=0.2, random_state=10)
    x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.25, random_state=20)
    return x_train, y_train, x_val, y_val, x_test, y_test
        

x_train, y_train, x_val, y_val, x_test, y_test = split_2D()

shape=(110, 110, 3)

model=MyModel(shape)


model.compile_and_fit(x_train,y_train,x_val,y_val,x_test,y_test)

