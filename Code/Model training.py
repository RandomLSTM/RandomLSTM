#for 70% data
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




