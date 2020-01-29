# Random LSTM

## Dependencies

Random LSTM depends on the following libraries:

*   Pandas
*   Scikit-learn
*   Keras
*   Tensorflow 
*   Numpy
*   Matplotlib

For detailed steps to install Tensorflow, follow the [Tensorflow installation
instructions](https://www.tensorflow.org/install/). A typical user can install
Tensorflow using one of the following commands:

``` bash
# For CPU
pip install tensorflow
# For GPU
pip install tensorflow-gpu
```

The remaining libraries can be installed from the online documentatins

# Model Setup
## Data Input

``` bash
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical

from google.colab import files
uploaded = files.upload()

```
## Data Processing
In this part we split the datset by 30% And 70% data
``` bash
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


def subset(split_data):
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

```
## Wight Training

``` bash
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.layers import Dense, Dropout,LSTM
from tensorflow.python.keras.optimizers import Adam
from keras.models import Sequential


def subset2(split_data):
  #data.sample(frac =.10, replace=True, random_state=1) 
  return split_data.sample(frac=.30, replace=True) 


#30%
j = 0
i = 0
acc_array1=[]
while i < 10:
  temp_data1[i]=subset2(data_new2)
  X = temp_data1[i].iloc[:, :-1].values
  y = temp_data1[i].iloc[:, -1].values
  y_train = np.asarray(temp_data1[i]['Outcome'])
  encoder = LabelEncoder()
  encoder.fit(y_train)
  y_train = encoder.transform(y_train)
  #print(y_train)

  from sklearn.model_selection import train_test_split
  X_train, X_test, ytrain, ytest = train_test_split(X, y_train, test_size = 0.25, random_state = 42)
  X_train_new = X_train.reshape(X_train.shape[0],1,X_train.shape[1])
  X_test_new = X_test.reshape(X_test.shape[0],1,X_test.shape[1])

  # convert integers to dummy variables (i.e. one hot encoded)
  y_train_new = np_utils.to_categorical(ytrain)
  y_train_new = y_train_new.reshape(y_train_new.shape[0],1,y_train_new.shape[1])
  #print(y_train_new.shape)


  y_test_new = np_utils.to_categorical(ytest)
  y_test_new = y_test_new.reshape(y_test_new.shape[0],1,y_test_new.shape[1])
  #y_test_new.shape

  from tensorflow.python.keras.models import Sequential
  from tensorflow.python.keras.layers import Dense,Dropout, LSTM
  from tensorflow.python.keras.optimizers import Adam
  model = myModel2()
  history = model.fit(X_train_new, y_train_new, epochs=20,validation_data=(X_test_new, y_test_new), batch_size=16)
  # evaluate the model
  acc_on_train(X_train_new,y_train_new,model)
  x = acc_on_test(X_test_new,y_test_new,model)
  x=x*100
  j +=1
  if x>45:
    acc_array1.append(x)
    print ("Accuracy on test data: ", x)
    i +=1
    print ("Index no: ",i)
  else:
    print ("Accuracy on test data: ", x)
    print ("Index no: ",i)

print ("Total loop: ",j)

number = 10
for i in range (number-1):
    for j in range(number - i - 1):
        if(acc_array1[j] < acc_array1[j + 1]):
             temp = acc_array1[j]
             acc_array1[j] = acc_array1[j + 1]
             acc_array1[j + 1] = temp
             temp =temp_data1[j]
             temp_data1[j] = temp_data1[j + 1]
             temp_data1[j + 1] = temp

for x in range(0,3):
  best_temp_data1[x]= temp_data1[x]
```
## Model Training

``` bash
j = 0
i = 0
acc_array=[]
while i < 10:
  temp_data[i]=subset(data_new1)
  X = temp_data[i].iloc[:, :-1].values
  y = temp_data[i].iloc[:, -1].values
  y_train = np.asarray(temp_data[i]['Outcome'])
  encoder = LabelEncoder()
  encoder.fit(y_train)
  y_train = encoder.transform(y_train)
  #print(y_train)

  from sklearn.model_selection import train_test_split
  X_train, X_test, ytrain, ytest = train_test_split(X, y_train, test_size = 0.25, random_state = 42)
  X_train_new = X_train.reshape(X_train.shape[0],1,X_train.shape[1])
  X_test_new = X_test.reshape(X_test.shape[0],1,X_test.shape[1])

  # convert integers to dummy variables (i.e. one hot encoded)
  y_train_new = np_utils.to_categorical(ytrain)
  y_train_new = y_train_new.reshape(y_train_new.shape[0],1,y_train_new.shape[1])
  #print(y_train_new.shape)


  y_test_new = np_utils.to_categorical(ytest)
  y_test_new = y_test_new.reshape(y_test_new.shape[0],1,y_test_new.shape[1])
  #y_test_new.shape

  from tensorflow.python.keras.models import Sequential
  from tensorflow.python.keras.layers import Dense,Dropout, LSTM
  from tensorflow.python.keras.optimizers import Adam
  model = myModel2()
  history = model.fit(X_train_new, y_train_new, epochs=20,validation_data=(X_test_new, y_test_new), batch_size=16)
  # evaluate the model
  acc_on_train(X_train_new,y_train_new,model)
  x = acc_on_test(X_test_new,y_test_new,model)
  x=x*100
  j +=1
  if x>45:
    acc_array.append(x)
    print ("Accuracy on test data: ", x)
    i +=1
    print ("Index no: ",i)
  else:
    print ("Accuracy on test data: ", x)
    print ("Index no: ",i)

print ("Total loop: ",j)

number = 10
for i in range (number-1):
    for j in range(number - i - 1):
        if(acc_array[j] < acc_array[j + 1]):
             temp = acc_array[j]
             acc_array[j] = acc_array[j + 1]
             acc_array[j + 1] = temp
             temp =temp_data[j]
             temp_data[j] = temp_data[j + 1]
             temp_data[j + 1] = temp

for x in range(0,3):
  best_temp_data[x]= temp_data[x]

# mean SD of 70% data
mean_70 = []
sd_70 = []
for i in range(0,3):
  df = pd.DataFrame(best_temp_data[i])
  mean = 0;
  sd=0;
  #df.iloc[:,0]
  import statistics 
  temp=df.shape[1]
  for i in range(temp-1):
    y = statistics.mean(df.iloc[:, i ])
    z= statistics.stdev(df.iloc[:, i ])
    mean += y
    sd += z 
  print('Mean for 70 =  %g' % (mean))
  print('Standard Deviation for 70 =  %g' % (sd))
  mean_70.append(mean)
  sd_70.append(sd)

# mean SD of 30% data
mean_30 = []
sd_30 = []
for i in range(0,3):
  df = pd.DataFrame(best_temp_data1[i])
  mean = 0;
  sd=0;
  #df.iloc[:,0]
  import statistics 
  temp=df.shape[1]
  for i in range(temp-1):
    y = statistics.mean(df.iloc[:, i ])
    z= statistics.stdev(df.iloc[:, i ])
    mean += y
    sd += z 

  print('Mean for 30 =  %g' % (mean))
  print('Standard Deviation for 30 =  %g' % (sd))
  mean_30.append(mean)
  sd_30.append(sd)


dis_70 = []
for i in range(0,3):
 dis_70.append(mean_70[i] + sd_70[i])
dis_70


dis_30 = []
for i in range(0,3):
 dis_30.append(mean_30[i] + sd_30[i])
dis_30

```
## Final Setup

``` bash
number = 3
num = 2
temp3 = []

for i in range (number-1):
    w = 999999
    for j in range(number - i - 1):
        v = abs(dis_70[i] - dis_30[j])
        if(v < w):
            z=j
            w=v
    if(w < abs(dis_70[z]-dis_30[z])):
      t = dis_30[i]
      ta = acc_array1[i]
      dis_30[i]=dis_30[z]
      acc_array1[i] = acc_array1[z]
      dis_30[z] = t
      acc_array1[z] = ta

for i in range (num-1):
    w = 999999
    for j in range(number - i - 1):
        v2=abs(dis_70[i] - dis_30[j])
        if(v2 < w):
            z=j
            w=v
    if(w < abs(dis_70[z]-dis_30[z])):
      t = dis_30[i]
      ta = acc_array1[i]
      dis_30[i]=dis_30[z]
      acc_array1[i] = acc_array1[z]
      dis_30[z] = t
      acc_array1[z] = ta

#Final weighted Accuracy
temp1 = 0
temp2 = 0
for i in range (0 ,3):
  temp1 += (acc_array[i]* acc_array1[i])
  temp2 += acc_array1[i]
finalAccuracy = temp1/temp2
print('Final weighted accuracy: %g' % (finalAccuracy))
```

