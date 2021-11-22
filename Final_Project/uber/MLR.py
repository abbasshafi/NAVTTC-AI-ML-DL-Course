import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv("taxi.csv")
# print(data)

x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values
# print(x)
# print(y)

x_train,y_train,x_test,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

reg= LinearRegression()
reg.fit(x_train,y_train)

print("Training Score",reg.score(x_train,y_train))
