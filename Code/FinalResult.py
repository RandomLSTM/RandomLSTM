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
