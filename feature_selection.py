import pandas as pd

path =r'c:\Users\INDOMITABLE ROCK\PycharmProjects\file_folder\sales_nonull.csv'
df = pd.read_csv(path)     
 df= df.drop(['items','Sub Category'],axis=1)
#print(df.shape)  # (1757, 5)
# print(df[0:10])  # show a snippet of the modified data frame
array = df.values
X = array[:,0:6]
#X = array[:,0:4]
#print(X)
Y = array[:,6]
# Y = array[:,4]
# print(Y)
#print(X.shape, Y.shape)  #(1757, 4) (1757,)


#we shall drop 'Rating','No_of_Review' cols as this was merged to give the outcome
df= df.drop(['items','Sub Category','Rating','No_of_Review'],axis=1)
#print(df.shape)  # (1757, 5)
# print(df[0:10])  # show a snippet of the modified data frame
array = df.values

X = array[:,0:4]
#print(X)
Y = array[:,4]
# print(Y)
#print(X.shape, Y.shape)  #(1757, 4) (1757,)
#DATA SELCTION
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
# configure to select all features
fs = SelectKBest(score_func=f_regression, k='all')
#k= all feature
# learn relationship from training data
fs.fit(X_train, y_train)
# transform train input data
X_train_fs = fs.transform(X_train)
# transform test input data
X_test_fs = fs.transform(X_test)
# what are scores for the features
fs.scores_
for i in range(len(fs.scores_)):
    print('Feature %d: %f' % (i, fs.scores_[i]))
#plot the scores
pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
pyplot.show()

