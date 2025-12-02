# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:

1. Import the necessary python packages using import statements.

2. Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().

3. Split the dataset using train_test_split.

4. Calculate Y_Pred and accuracy.

5. Print all the outputs.

6. End the Program.


## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Muhammad Afshan A
RegisterNumber:  212223100035
*/

import chardet
file='spam.csv'
with open(file,'rb') as rawdata:
    result=chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv('spam.csv',encoding='Windows-1252')
print("Name:  Muhammad Afshan A")
print("Reg No: 212223100035")
data.head()
data.isnull().sum()
data.info()
x=data["v2"].values
y=data["v1"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
x_train
x_test
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
cm=metrics.confusion_matrix(y_test,y_pred)
cm
cf=metrics.classification_report(y_test,y_pred)
cf
```

## Output:

![alt text](image.png)

![alt text](image-1.png)

![alt text](image-2.png)

![alt text](image-3.png)

![alt text](image-4.png)

![alt text](image-5.png)

![alt text](image-6.png)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
