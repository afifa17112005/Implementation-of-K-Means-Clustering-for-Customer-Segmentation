# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:

1. Import the standard libraries.
2. Upload the dataset and check for any null values using .isnull() function. 
3. Import LabelEncoder and encode the dataset.
4. Import DecisionTreeRegressor from sklearn and apply the model on the dataset.
5. Predict the values of arrays.
6. Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.
7. Predict the values of array.
8. Apply to new unknown values

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: AFIFA A
RegisterNumber: 212223040008
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
data = pd.read_csv("Salary_EX7.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
y=data["Salary"]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor,plot_tree
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse = metrics.mean_squared_error(y_test,y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict(pd.DataFrame([[5,6]], columns=x.columns))
plt.figure(figsize=(20, 8))
plot_tree(dt, feature_names=list(x.columns), filled=True)
plt.show() 
*/
```

## Output:

## Data Head:

<img width="486" height="366" alt="Screenshot 2025-10-25 132621" src="https://github.com/user-attachments/assets/5f0b29e0-00f8-43bb-b8bb-b0efb09bdce3" />

## Data Info :

<img width="475" height="300" alt="Screenshot 2025-10-25 132638" src="https://github.com/user-attachments/assets/9467f515-3d6c-4ff1-9117-f32db8d2a6a6" />

## Data Details :

<img width="657" height="774" alt="Screenshot 2025-10-25 132705" src="https://github.com/user-attachments/assets/93ac721b-8c48-4c4f-9952-6dceb3970894" />

## Data Predcition :

<img width="1034" height="391" alt="Screenshot 2025-10-25 132719" src="https://github.com/user-attachments/assets/b96f5940-1f5c-44c7-9e8c-7390aeb5f416" />

## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
