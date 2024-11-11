# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas,scikit-learn for data handling and SVC for building the SVM model.
2. Load the dataset, clean the text, and vectorize it using TfidfVectorizer to convert emails into numerical features.
3. Split the dataset into training and testing sets using train_test_split.
4. Train an SVM model with a linear kernel (SVC) on the training data.
5. Evaluate the model by predicting on the test set and calculating metrics like accuracy, precision, recall, and F1-score.

## Program:
```

Developed by: T.Roshini
RegisterNumber: 212223230175

import chardet
file="spam.csv"
with open(file,'rb') as rawdata:
    result=chardet.detect(rawdata.read(100000))
result

import pandas as pd
df=pd.read_csv("C:/Users/admin/Downloads/spam.csv",encoding= 'Windows-1252')
df

df.head()

df.info()

df.isnull().sum()

x=df['v2'].values
y=df['v1'].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
#CountVectorizer is convert text into numerical data

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:

#### result
![image](https://github.com/user-attachments/assets/a6bb8afa-362e-47e2-bbbf-15812ac673cc)

#### Data
![image](https://github.com/user-attachments/assets/1ce543ba-391d-4de5-a756-b369040ba665)

#### Head
![image](https://github.com/user-attachments/assets/98e26c90-87d8-4429-9195-57a841d1c694)

#### Info
![image](https://github.com/user-attachments/assets/f6f7483f-59e0-4b43-bce7-fe7fb3b60fa4)

#### Null.sum
![image](https://github.com/user-attachments/assets/14f385d6-bfc2-4fd1-a42f-d0e2ff03fb23)

#### y_pred
![image](https://github.com/user-attachments/assets/952d2f03-a113-4fb2-a20d-24646bb5213c)

#### accuracy
![image](https://github.com/user-attachments/assets/bcd58ff8-960a-482f-b5b5-47b35926e4e5) 

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
