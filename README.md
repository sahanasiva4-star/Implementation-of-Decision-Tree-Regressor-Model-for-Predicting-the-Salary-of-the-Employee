# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the libraries and read the data frame using pandas.
2.Calculate the null values present in the dataset and apply label encoder.
3.Determine test and training data set and apply decison tree regression in dataset.
4.Calculate Mean square error,data prediction and r2.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Sahana S
RegisterNumber: 25013621
*/
```
~~~
import pandas as pd
df=pd.read_csv("Salary.csv")
df.head()
df.info()
df.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df["Position"]=le.fit_transform(df["Position"])
print(df.head())

x=df[["Position","Level"]]
y=df["Salary"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()

#Y_pred
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
print(y_pred)

#MSE
from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse

#RMSE
import numpy as np
rmse=np.sqrt(mse)
rmse

#ACCURACY
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
~~~

## Output:
<img width="260" height="215" alt="image" src="https://github.com/user-attachments/assets/0f19640a-e78d-455e-a563-dcb5c6385784" />
<img width="163" height="91" alt="image" src="https://github.com/user-attachments/assets/971ae1a6-5f34-4f93-9d80-574235ffd9de" />
#Y_pred
<img width="200" height="34" alt="image" src="https://github.com/user-attachments/assets/34e0d78b-4692-40d3-9719-d87898625fc1" />
#MSE
<img width="160" height="40" alt="image" src="https://github.com/user-attachments/assets/a1cc2044-3252-4ad5-ba0d-7c618982993e" />
#RMSE
<img width="248" height="34" alt="image" src="https://github.com/user-attachments/assets/34e6a298-ace4-45dc-992b-ccbb069a20a4" />
#ACCURACY 
<img width="239" height="31" alt="image" src="https://github.com/user-attachments/assets/bc16e9b5-9b1a-464b-9db2-0d4fd10b73cb" />


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
