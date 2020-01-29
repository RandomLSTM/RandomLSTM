import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical


from google.colab import files
uploaded = files.upload()





import pandas as pd
data=pd.read_csv("diabetes.csv")
data.head()

split_count=data.shape[0]*0.7
split_count=int(split_count)
data_new1= data.iloc[:split_count, :]
data_new2= data.iloc[split_count:, :]


import array as arr
#data = arr.array('d', 10)
temp_data = [None] * 10
temp_data1 = [None] * 10
best_temp_data = [None] * 3
best_temp_data1 = [None] * 3
#for i in range(0,9):
    #  data[i]=df.sample(frac=0.1, replace=True, random_state=1)
    #temp_data[i]=data.sample(frac =.10,replace=True) 
    #print(temp_data[i])

def subset(split_data):
  #data.sample(frac =.10, replace=True, random_state=1) 
  return split_data.sample(frac=.10, replace=True) 

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

#for encoding
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

def myModel2():
  model = Sequential()
  model.add(LSTM(128,input_shape=(1,X_train_new.shape[2]), return_sequences=True))
  model.add(LSTM(64,return_sequences=True))
  model.add(LSTM(16,return_sequences=True))
  model.add(LSTM(8, activation='relu', return_sequences=True))
  model.add(Dense(2,activation='softmax'))
  optimizer = Adam(lr=.001)
  model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy', 'mae','mse'])
  model.summary()
  return model

def plot_accuracy_loss(history):
  history_dict = history.history
  # plot loss during training
  loss_values = history_dict['loss']
  val_loss_values = history_dict['val_loss']
  plt.subplot(211)
  plt.title('Loss')
  epochs = range(1, len(loss_values) + 1)
  plt.plot(epochs,loss_values, label='Training loss')
  plt.plot(epochs,val_loss_values, label='test/Validation loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()
  # plot accuracy during training
  acc_values = history_dict['acc']
  val_acc_values = history_dict['val_acc']
  plt.subplot(212)
  plt.title('Accuracy')
  epochs = range(1, len(acc_values) + 1)
  plt.plot(epochs,acc_values, label='train accuracy')
  plt.plot(epochs,val_acc_values, label='test/Validation accuracy')
  plt.legend()
  plt.show()
  
def acc_on_train(x,y,model):
  %%time
  result = model.evaluate(x, y,verbose=0)
  print("Accuracy on trian data: {0:.2%}".format(result[1]))
  print("MAE on train data: {0:.2%}".format(result[2]))
  print("MSE on train data: {0:.2%}".format(result[3]))
  print("\n")
  
def acc_on_test(x,y,model):
  %%time
  result = model.evaluate(x, y,verbose=0)
  print("Accuracy on test data: {0:.2%}".format(result[1]))
  print("MAE on test data: {0:.2%}".format(result[2]))
  print("MSE on test data: {0:.2%}".format(result[3]))
  print("\n")
  return result[1]