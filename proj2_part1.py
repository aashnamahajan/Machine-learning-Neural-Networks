
# coding: utf-8

# In[1]:


import util_mnist_reader as mnist_reader
import numpy as np
from numpy import exp,reshape
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# In[2]:


#sigmoid function
def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s


# In[3]:


#calculating_loss
def calc_loss(Y, a2):
    loss_sum = np.sum(np.multiply(Y, np.log(a2)))              
    m = Y.shape[1]
    loss = -(1/m) * loss_sum
    return loss


# In[4]:


#calculating softmax function
def softmax_function(z2):
    a2 = np.exp(z2) / np.sum(np.exp(z2), axis=0)                 # (60000,10)
    return a2


# In[5]:


def function_part_1():
    #loading the data
    X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
    X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

    #spliting the test data into test data and val data
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, shuffle=False)

    #normalize the data
    X_train = X_train / 255
    X_test = X_test / 255
    X_val = X_val / 255

    digits = 10               # no of classes 
    examples = y_train.shape[0]         

    y_train = y_train.reshape(1, examples)

    Y_new = np.eye(digits)[y_train.astype('int32')]
    y_train = Y_new.T.reshape(digits, examples)

    examples_val = y_val.shape[0] 
    y_val = y_val.reshape(1, examples_val)

    Y_val_new = np.eye(digits)[y_val.astype('int32')]
    y_val = Y_val_new.T.reshape(digits, examples_val)

    m=60000
    n_x = X_train.shape[1]                              # 784

    n_h = 64                                            # no_of_nodes
    learning_rate = 1

    W1 = np.random.randn(n_h, n_x)
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(10, n_h)
    b2 = np.zeros((10, 1))

    X = X_train
    Y = y_train

    loss_list=[]
    loss_val_list=[]

    for i in range(1500):

        #training dataset

        Z1 = np.matmul(W1, X.T) + b1             # (64, 60000)
        A1 = sigmoid(Z1)
        Z2 = np.matmul(W2, A1) + b2
        A2 = softmax_function(Z2)

        cost = calc_loss(Y, A2)
        loss_list.append(cost)
        
        #validation dataset

        Z1_val = np.matmul(W1, X_val.T) + b1             # (64, 60000)
        A1_val = sigmoid(Z1_val)
        Z2_val = np.matmul(W2, A1_val) + b2
        A2_val = softmax_function(Z2_val)

        cost_val = calc_loss(y_val, A2_val)
        loss_val_list.append(cost_val)

        dZ2 = A2-Y
        dW2 = (1./m) * np.matmul(dZ2, A1.T)
        db2 = (1./m) * np.sum(dZ2, axis=1, keepdims=True)

        dA1 = np.matmul(W2.T, dZ2)
        dZ1 = dA1 * sigmoid(Z1) * (1 - sigmoid(Z1))
        dW1 = (1./m) * np.matmul(dZ1, X)
        db1 = (1./m) * np.sum(dZ1, axis=1, keepdims=True)

        W2 = W2 - learning_rate * dW2
        b2 = b2 - learning_rate * db2
        W1 = W1 - learning_rate * dW1
        b1 = b1 - learning_rate * db1

        if i % 100 == 0:
            print("Epoch", i, "cost: ", cost)

    print("Final cost:", cost)

    #plotting the data
    #%matplotlib inline
    get_ipython().run_line_magic('matplotlib', 'inline')
    plt.plot(loss_list)
    plt.plot(loss_val_list)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title('Loss versus Epoch graph for Single layer neural network')
    plt.legend(["Training set","Validation set"])

    plt.show()
    
    Z1 = np.matmul(W1,X_test.T) + b1
    A1 = sigmoid(Z1)
    Z2 = np.matmul(W2,A1) + b2
    A2 = softmax_function(Z2)

    predictions = np.argmax(A2, axis=0)
    # labels = np.argmax(y_test, axis=0)
    print(confusion_matrix(predictions, y_test))
    print(classification_report(predictions, y_test))

