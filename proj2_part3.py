
# coding: utf-8

# In[1]:


from matplotlib import pyplot
from keras.datasets import fashion_mnist
import util_mnist_reader as mnist_reader
import keras
import keras.utils
from keras import utils as np_utils
from sklearn.model_selection import KFold
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.optimizers import SGD
import statistics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


# In[2]:


def function_part3():
    #load data
    X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
    X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, shuffle=False) 

    Y=y_test

    #reshape data
    X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
    X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))
    X_val = X_val.reshape((X_val.shape[0],28,28,1))

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    y_val = to_categorical(y_val)
    
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_val = X_val.astype('float32')

    # normalize to range 0-1
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    X_val = X_val / 255.0
    
    #defining the model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0,2))
    model.add(Dense(10, activation='softmax'))

    #Model configuration
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    
    #Model training
    nepochs = 5
    history = model.fit(X_train, y_train, epochs=nepochs, batch_size=36, validation_data=(X_val, y_val))
    
    # evaluate model
    test_loss,  test_accuracy = model.evaluate(X_test, y_test, verbose=2)

    #plotting the loss versus accuracy graph
    get_ipython().run_line_magic('matplotlib', 'inline')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title('Loss versus Epoch graph for convolutional neural network')
    plt.legend(["Training set","Validation set"])
    plt.plot(range(nepochs), history.history['loss'], 'y', range(nepochs), history.history['val_loss'], 'g')

    plt.show()
    
    predictions = model.predict(X_test)
    np_predictions = np.argmax(predictions,axis = 1)
    print(confusion_matrix(np_predictions, Y))
    print(classification_report(np_predictions, Y))


# In[3]:




