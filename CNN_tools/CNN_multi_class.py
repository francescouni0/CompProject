import sys
from pathlib import Path
import os

sys.path.insert(0, str(Path(os.getcwd()).parent))

import numpy as np
import scipy
import matplotlib.pyplot as plt
import tensorflow
from sklearn.metrics import roc_curve, auc
from keras.layers import BatchNormalization, Dense, Flatten, Activation, Dropout, Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.losses import Hinge


class MyModel(tensorflow.keras.Model):
    """
    Subclass of the TensorFlow Keras Model class. This model is a Convolutional Neural Network (CNN) 2.5D designed for
    image classification tasks: it consists of two 2D convolutional layers followed by two fully connected layers,
    including batch normalization, activation functions, max pooling and dropout to improve its performance and
    generalization power.

    Attributes
    ----------
    conv1 : tensorflow.python.keras.layers.Conv2D
        First convolutional layer with 3x3 kernel size and 8 filters.
    bn1 : tensorflow.python.keras.layers.BatchNormalization
        First batch normalization layer.
    act1 : tensorflow.python.keras.layers.Activation
        First activation function layer with a Rectified Linear Unit function.
    pool1 : tensorflow.python.keras.layers.MaxPooling2D
        First max pooling layer with 2x2 pool size and stride 3.
    drop1 : tensorflow.python.keras.layers.Dropout
        First dropout layer with a 0.3 rate.
    conv2 : tensorflow.python.keras.layers.Conv2D
        Second convolutional layer with 3x3 kernel size and 16 filters.
    bn2 : tensorflow.python.keras.layers.BatchNormalization
        Second batch normalization layer.
    act2 : tensorflow.python.keras.layers.Activation
        Second activation function layer with a Rectified Linear Unit function.
    pool2 : tensorflow.python.keras.layers.MaxPooling2D
        Second max pooling layer with 3x3 pool size and stride 3.
    flatten : tensorflow.python.keras.layers.Flatten
        Flatten layer.
    fc1 : tensorflow.python.keras.layers.Dense
        First fully connected layer with 116 neurons and ReLU activation function.
    fc2 : tensorflow.python.keras.layers.Dense
        Second fully connected layer with a single neuron and sigmoid activation function for binary classification.
    """
    def __init__(self, shape=(110, 110, 3)):
        """
        Initializes the "MyModel" object with the given shape of the input images (default = (110, 110, 3)). It creates
        the model architecture by defining the layers and the associated parameters.

        Parameters
        ----------
        shape : tuple
            Tuple containing the (x,y) dimensions of image matrix and the number of slices (or channels). Default is
            (110. 110, 3).

        Returns
        -------
            None
        """
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(8, (3, 3), padding='valid', input_shape=shape, kernel_regularizer='l1')
        self.bn1 = BatchNormalization()
        self.act1 = Activation('relu')
        self.pool1 = MaxPooling2D((2, 2), strides=3)
        self.drop1 = Dropout(0.3)
        self.conv2 = Conv2D(16, (3, 3), padding='valid', input_shape=shape, kernel_regularizer='l1')
        self.bn2 = BatchNormalization()
        self.act2 = Activation('relu')
        self.pool2 = MaxPooling2D((3, 3), strides=3)
        self.flatten = Flatten()
        self.fc1 = Dense(116, activation='relu')
        self.fc2 = Dense(1, activation='sigmoid')

    def call(self, inputs):
        """
        Implements the forward pass of the model, taking the input images and feeding them through each layer of the CNN.
        The order of execution is as specified in the constructor.

        Parameters
        ----------
        inputs : numpy.ndarray
            Input DTI images to the CNN.

        Returns
        -------
        self.fc2 : numpy.ndarray
            A numpy array of shape (n_samples, 1) representing binary classification prediction for each input sample,
            where n_samples is the number of input images.
        """
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
    
    def compile_and_fit(self, x_train, y_train, x_val, y_val, x_test, y_test, n_epochs, batchsize):
        """
        Compiles the model and fits it to the training data. Once fitted, the training process is shown by loss and
        accuracy plots (both for training and validation) and the overall performance is evaluated on the validation and
        test set.

        Parameters
        ----------
        x_train : numpy.ndarray
            Train subset of DTI images.
        y_train : numpy.ndarray
            Labels corresponding to x_train.
        x_val : ndarray
            Validation subset of DTI images.
        y_val : numpy.ndarray
            Labels corresponding to x_val.
        x_test : numpy.ndarray
            Test subset of DTI images.
        y_test : numpy.ndarray
            Labels corresponding to x_test.
        n_epochs : int
            Maximum number of epochs to train the models.
        batchsize : int
            Batch size to use during training.

        Returns
        -------
            None
        """
        self.compile(optimizer=SGD(learning_rate=0.01), loss=Hinge(), metrics=['accuracy'])

        reduce_on_plateau = ReduceLROnPlateau(monitor="val_loss",
                                              factor=0.1,
                                              patience=20,
                                              verbose=0,
                                              mode="auto",
                                              min_delta=0.0001,
                                              cooldown=0,
                                              min_lr=0)

        early_stopping = EarlyStopping(monitor="val_loss",
                                       min_delta=0,
                                       patience=20,
                                       verbose=0,
                                       mode="auto",
                                       baseline=None,
                                       restore_best_weights=False,
                                       start_from_epoch=10)

        epochs = n_epochs
        batch_size = batchsize

        history = self.fit(x_train, y_train,
                           batch_size=batch_size,
                           epochs=epochs,
                           steps_per_epoch=round(len(x_train)/batch_size),
                           verbose=1,
                           validation_data=(x_val, y_val),
                           validation_steps=round(len(x_val)/batch_size),
                           callbacks=[reduce_on_plateau, early_stopping])

        self.accuracy_loss_plot(history)
        self.validation_roc(x_val, y_val)
        self.test_roc(x_test, y_test)
        self.save_weights(Path('model.h5'))

    def accuracy_loss_plot(self, history):
        """
        Generates two plots: one of model's accuracy on the training and validation sets, the other of model's loss on
        the same sets of data.

        Parameters
        ----------
        history : keras.callbacks.History
            History object generated during model training.

        Returns
        -------
            None
        """
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(1, len(acc)+1)

        # Train and validation accuracy
        plt.figure(figsize=(15, 15))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        # Train and validation loss
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()
        
    def validation_roc(self, x_val, y_val):
        """
        Generates a plot of the ROC curve and calculates the AUC score for the model on the validation set.

        Parameters
        ----------
        x_val : ndarray
            Validation subset of DTI images.
        y_val : numpy.ndarray
            Labels corresponding to x_val.

        Returns
        -------
            None
        """
        confidence_int = 0.683
        z_score = scipy.stats.norm.ppf((1 + confidence_int) / 2.0)

        _, val_acc = self.evaluate(x_val, y_val, verbose=0)
        accuracy_err = z_score * np.sqrt((val_acc * (1 - val_acc)) / y_val.shape[0])
        print('Validation accuracy:', round(val_acc, 2), "+/-", round(accuracy_err, 2))
    
        preds = self.predict(x_val, verbose=1)

        # Compute Receiver operating characteristic (ROC) curve
        fpr, tpr, _ = roc_curve(y_val, preds)
        roc_auc = auc(fpr, tpr)
        n1 = sum(y_val == 1)
        n2 = sum(y_val == 0)
        q1 = roc_auc / (2 - roc_auc)
        q2 = 2 * roc_auc ** 2 / (1 + roc_auc)
        auc_err = z_score * np.sqrt(
            (roc_auc * (1 - roc_auc) + (n1 - 1) * (q1 - roc_auc ** 2) + (n2 - 1) * (q2 - roc_auc ** 2)) / (n1 * n2)
        )
        print("AUC:", round(roc_auc, 2), "+/-", round(auc_err, 2))
    
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
        
    def test_roc(self, x_test, y_test):
        """
        Generates a plot of the ROC curve and calculates the AUC score for the model on the test set.

        Parameters
        ----------
        x_test : ndarray
            Test subset of DTI images.
        y_test : numpy.ndarray
            Labels corresponding to x_test.

        Returns
        -------
            None
        """
        confidence_int = 0.683
        z_score = scipy.stats.norm.ppf((1 + confidence_int) / 2.0)

        test_loss, test_acc = self.evaluate(x_test, y_test)
        accuracy_err = z_score * np.sqrt((test_acc * (1 - test_acc)) / y_test.shape[0])
        print('Test accuracy:', round(test_acc, 2), "+/-", round(accuracy_err, 2))
        
        preds_test = self.predict(x_test, verbose=1)

        # Compute Receiver operating characteristic (ROC) curve
        fpr, tpr, _ = roc_curve(y_test, preds_test)
        roc_auc = auc(fpr, tpr)
        n1 = sum(y_test == 1)
        n2 = sum(y_test == 0)
        q1 = roc_auc / (2 - roc_auc)
        q2 = 2 * roc_auc ** 2 / (1 + roc_auc)
        auc_err = z_score * np.sqrt(
            (roc_auc * (1 - roc_auc) + (n1 - 1) * (q1 - roc_auc ** 2) + (n2 - 1) * (q2 - roc_auc ** 2)) / (n1 * n2)
        )
        print("AUC:", round(roc_auc, 2), "+/-", round(auc_err, 2))

        # Plot of a ROC curve
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
    
    def load(self, path, x_train, y_train, x_val, y_val, x_test, y_test, n_epochs, batchsize):
        """
        Loads, compiles and re-trains a previously trained model stored in a H5 file.

        Parameters
        ----------
        path : str
            Path to the H5 file storing the pre-trained model.
        x_train : numpy.ndarray
            Train subset of DTI images.
        y_train : numpy.ndarray
            Labels corresponding to x_train.
        x_val : ndarray
            Validation subset of DTI images.
        y_val : numpy.ndarray
            Labels corresponding to x_val.
        x_test : numpy.ndarray
            Test subset of DTI images.
        y_test : numpy.ndarray
            Labels corresponding to x_test.
        n_epochs : int
            Maximum number of epochs to train the models.
        batchsize : int
            Batch size to use during training.

        Returns
        -------
            None
        """
        self.compile(optimizer=SGD(learning_rate=0.01), loss=Hinge(), metrics=['accuracy'])
        self.train_on_batch(x_train, y_train)
        self.load_weights(path)
        self.compile_and_fit(x_train, y_train, x_val, y_val, x_test, y_test, n_epochs, batchsize)
