import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

dataset = pd.read_csv(r'C:\Users\HP\OneDrive\Desktop\Documents\Practicing\Medical Insurance Cost Prediction\medical-charges.csv')

x = dataset.drop('charges',axis = 1)
y = dataset['charges']


categorical_cols = ['sex','smoker','region']
numerical_cols = ['age','bmi','children']

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('cat',OneHotEncoder(drop='first',handle_unknown='ignore'),categorical_cols),
        ('num','passthrough',numerical_cols)
        ]
    )

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
# model = Pipeline(steps=[
#     ('preprocessing',preprocessor),
#     ('regressor' , LinearRegression())
#     ])

# split 
from sklearn.model_selection import train_test_split
x_train ,x_test ,y_train ,y_test = train_test_split(x,y, test_size=0.25,random_state=0)

#model.fit(x_train,y_train)

#y_pred = model.predict(x_test)

#========= try with random forest
from sklearn.ensemble import RandomForestRegressor
rf = Pipeline(steps=[
    ('preprocessing',preprocessor),
    ('regression',RandomForestRegressor(n_estimators=200,random_state=42,n_jobs=-1))
    ])

rf.fit(x_train, y_train)
y_pred_rf = rf.predict(x_test)

from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
mse = mean_squared_error(y_test, y_pred_rf)
print(mse)

mae = mean_absolute_error(y_test, y_pred_rf)
print(mae)

r2 = r2_score(y_test, y_pred_rf)
print(r2)
