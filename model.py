import pandas as pd
from  matplotlib import pyplot
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold,RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler,MinMaxScaler,Normalizer
from sklearn.tree import DecisionTreeRegressor
import numpy

#predict which product will sell faster
#we shall  rate product prospects as positive reviews X high numb of reveiws
#predict the perceived appeal rating of an item by customers
#we shall  rate perceived appea of items  as positive reviews X high number of reveiws
# from the grocery dataset downloade from kaggle.com amd removed the null values
# and saved as sales_nonull.csv
#the perceived appeal rating is a regression problem
#regression problem  poluarity is based on non catgorical numbers

#open and read file
path =r'c:\Users\INDOMITABLE ROCK\PycharmProjects\file_folder\sales_nonull.csv'
df = pd.read_csv(path)



# # #drop text cols or unsued cols rating and reviews has been marged to outcome
# # #high corelation or redundancy can affect outome
#,'Rating','No_of_Review'
df= df.drop(['items','Sub Category','Rating','No_of_Review'],axis=1)
#print(df.shape)
print(df[0:10])  # show a snippet of the modified data frame
array = df.values

X = array[:,0:4]
#print(X)
Y = array[:,4]
# print(Y)
#print(X.shape, Y.shape)  #(1757, 4) (1757,)

#
# MODELING

model= DecisionTreeRegressor()
kfold = KFold(n_splits=10)
pipeModel = Pipeline([('scaler',StandardScaler()),("DecisionTreeRegressor", model)])
cv_results = cross_val_score(pipeModel, X, Y, cv=kfold)
print(cv_results.mean())
pipeModel.fit(X,Y)  # model is inside the pipemodel
scores1= pipeModel.score(X,Y)
print('score=',scores1)
model_predict = pipeModel.predict(X)  #model.predict() gives the y prediction
from sklearn.metrics import r2_score
scores3= r2_score( Y,model_predict)  #score the predciton with actaul y values
print("Model accucarcy=",scores3)


#evaluation of the model: give it arbitary figure
input_data=  [1,1,56.99,0]  #
#convert list ot array
import numpy
Input_array = numpy.array(input_data)
print('Input_array.shape:',Input_array.shape)  #(4,) (rows=4,cols=0)
#the input_data  must be in same shape as X = array[:,0:4]
print('X.shape',X.shape)
#so we need to reshape Input_array from (4,) (rows=4,cols=0) to be like X_array (1row,4cols)
Input_array_reshaped = Input_array.reshape(1,-1)
print('Input_array_reshaped.shape:',Input_array_reshaped.shape) #(1, 4) instead of  (4,)
make_prediction = pipeModel.predict(Input_array_reshaped)
print(make_prediction)



print(numpy.percentile(df['BuyingProspect'],75))
print(numpy.percentile(df['BuyingProspect'],0))
a= make_prediction-numpy.percentile(df['BuyingProspect'],0)
print(a)
b = numpy.percentile(df['BuyingProspect'],98)- numpy.percentile(df['BuyingProspect'],0)
print(b)
perc = a/b *100
print("selling prospect:%s"%perc,"%")

#
#sve your model
import pickle
filename = 'trained_pipeModel.sav'
#RUN THIS PYTHON FILE CREATE THIS FILE 'trained_pipeModel.sav'
pickle.dump(pipeModel,open(filename, 'wb'))
#
# #load the saved model.
loaded_model = pickle.load(open('trained_pipeModel.sav','rb'))
# #creat an interface
# #NEXT STEP ON TIHS use trained_lrmodel.sav on sellingProspectAPP.py
# #configure to webrowser using STreamlit
# #to run the application on web browser ...
# # #run  on your pycharm terminal.... ' streamlit run microwaveAPP.py '






