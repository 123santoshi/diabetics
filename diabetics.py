

**INSTALL NECESSARY LIBRARIES**


!pip install numpy
!pip install pandas
!pip install scikit-learn

"""**IMPORTING LIBRARIES**"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import train_test_split

"""ADDING LOCAL FILE"""

from google.colab import files
d=files.upload()

"""LOAD THE DATASET """

data=pd.read_csv("/content/diabetes.csv")
data

"""**SUMMARIZE THE DATASET**"""

data.head()             #it displays top 5 rows in the dataset

data.describe()     #it gives description about dataset

data.tail()         #it displays last 5 rows in the dataset

data.shape          #it shows(no of rows,no of columns)

"""**SEGREGEATE DATA INTO INPUT AND OUPUT**"""

x=data.drop(columns="diabetes")         #input data
x

y=data.diabetes                       #output label
y

"""PREPROCESS THE INPUT(NORMALIZATION)"""

scale=StandardScaler()
scale

x_scale=scale.fit_transform(x)
x_scale

"""**TRAIN THE MODEL USING SVM MODEL**"""

x_train,x_test,y_train,y_test=train_test_split(x_scale,y,test_size=0.2)

x_train.shape
x_test.shape

svmclassifier=svm.SVC(kernel="linear")
svmclassifier.fit(x_train, y_train)

"""**FINDING ACCURACY OF TRAINING DATA,TEST DATA**"""

# accuracy score on the training data
x_train_prediction = svmclassifier.predict(x_train)
training_accuracy = accuracy_score(x_train_prediction, y_train)
accuracy=100*training_accuracy
print("accuracy of trianing data={0}".format(accuracy))

# accuracy score on the training data
x_test_prediction = svmclassifier.predict(x_test)
testing_accuracy = accuracy_score(x_test_prediction, y_test)
accuracy=100*testing_accuracy
print("accuracy of testing  data={0}".format(accuracy))

"""**PREDICTING /TESTING THE DATA**"""

#input= (5,166,72,19,175,25.8,0.587,51,1.09)
input=(6,148,72,35,0,33.6,0.627,50,1.3790)
input_array = np.asarray(input)
input_reshape = input_array.reshape(1,-1)
std_data = scale.transform(input_reshape)
print(std_data)
prediction = svmclassifier.predict(std_data)
print(prediction)

"""**CHECK WHETHER PERSON HAVE DIABETICS OR NOT**"""

if (prediction[0] == 0):
  print('The person is not  diabetic')
else:
  print('The person is diabetic')