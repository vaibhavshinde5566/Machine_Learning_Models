#Simple machine learning
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

dataset = pd.read_csv(r'C:\Users\HP\OneDrive\Desktop\Documents\Machine Learning programs\Salary_Data.csv')
x = dataset.iloc[:, 0].values.reshape(-1, 1) 
y = dataset.iloc[:,1].values

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)

plt.scatter(x, y, color = 'red')
plt.plot(x,y,color = 'blue')
plt.xlabel('Exp')
plt.ylabel('Salary')

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2,random_state=0)

m = lin_reg.coef_
print(m)

c = lin_reg.intercept_
print(c)

y_pred = lin_reg.predict([[2.5]])
print(y_pred)

plt.scatter(x, y, color = 'red')
plt.plot(x,lin_reg.predict(x),color = 'blue')
plt.xlabel('Exp')
plt.ylabel('Salary')




with open(r"C:\Users\HP\OneDrive\Desktop\salary_model.pkl", "wb") as file:
    pickle.dump(lin_reg, file)

print("Model saved on Desktop")