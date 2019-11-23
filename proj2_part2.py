
# coding: utf-8

# In[1]:


import util_mnist_reader as mnist_reader
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import train_test_split


# In[2]:


def function_part2():
    
    #data loading
    X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
    X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, shuffle=False) 

    Y=y_test

    #convert data to one_hot representation
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)
    y_val = tf.keras.utils.to_categorical(y_val)
    
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    #reshaping the dataset
    X_train = X_train.reshape(60000,28,28)
    X_val = X_val.reshape(5000,28,28)
    X_test = X_test.reshape(5000,28,28)
    
    #normalizing the datset to get it in a range of 0 and 1
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    X_val = X_val / 255.0
    
    #defining the model
    model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(198, activation='sigmoid'),
    keras.layers.Dense(88, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
    ])

    #Model configuration
    model.compile(optimizer='SGD',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    
    train_loss = model.fit(X_train, y_train, validation_data= (X_val, y_val), epochs=80)
    
    train_loss_list = train_loss.history['loss']
    val_loss_list = train_loss.history['val_loss']
    accuracy_list= train_loss.history['accuracy']
    val_accuracy_list= train_loss.history['val_accuracy']

    #plotting the loss versus accuracy graph
    get_ipython().run_line_magic('matplotlib', 'inline')
    plt.plot(train_loss_list)
    plt.plot(val_loss_list)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title('Loss versus Epoch graph for Multi layer neural network')
    plt.legend(["Training set","Validation set"])
    
    plt.show()
    
    test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=1)

    print('\nTest accuracy:', test_acc)
    
    
    predictions = model.predict(X_test)
    np_predictions = np.argmax(predictions,axis = 1)
    print(confusion_matrix(np_predictions, Y))
    print(classification_report(np_predictions, Y))


# In[8]:





# In[10]:





# In[11]:





# In[12]:





# In[14]:




