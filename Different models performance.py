import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressorfrom sklearn.tree import DecisionTreeRegressorfrom sklearn.tree import ExtraTreeRegressorfrom sklearn.svm import SVR

#open and read file
path =r'c:\Users\INDOMITABLE ROCK\PycharmProjects\file_folder\sales_nonull.csv'
df = pd.read_csv(path)
#df= df.drop(['items','Items_id','Sub Category','Items_Category','Discount','Rating','No_of_Review',],axis=1)
array = df[['Price','BuyingProspect']].values  #select othe feature v outcome
 
X = array[:,0:1]  #df[['Price']
#print(X
y= array[:,1] #df['BuyingProspect']

# create a model try others models to see how they predict the data 
model = SVR(kernel='sigmoid')
 # model= DecisionTreeRegressor()# model= ExtraTreeRegressor()
# model = KNeighborsRegressor(n_neighbors=k)  where k= 1 to 21 
# Model= SVR(C=i) where c= 0.1 to 1
# train the model on the data
model.fit(X, y)
# make predictions on the data
y_pred = model.predict(X)
# plot the predicted values against the true values
#price v buying prospect scatter
plt.scatter(X, y, color='darkorange',label='data')
#price  v buying prospect prediction scatter
plt.plot(X, y_pred, color='cornflowerblue',label='prediction')
plt.legend()
plt.ylabel('buying prospect')
plt.xlabel('price in $')
plt.show()
