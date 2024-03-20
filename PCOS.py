
import pandas as pd
import pickle
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from matplotlib import pyplot as plt
from sklearn.svm import SVC

from google.colab import files
uploaded = files.upload()

df = pd.read_csv("PCOS_infertility.csv")
df = df.fillna(0)

df.columns

df

df.shape

"""df.describe"""

df['PCOS (Y/N)'].value_counts()

df.groupby('PCOS (Y/N)').mean()

X = df.drop(columns =['PCOS (Y/N)', 'Sl. No',  'Patient File No.'] )
Y = df['PCOS (Y/N)']

print(X)

print(Y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=1)

print(X.shape, X_train.shape, X_test.shape)
classifier = svm.SVC(kernel='linear')

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Step 2: Instantiate the SVC classifier
classifier = SVC()

# Step 3: Fit the classifier to the training data
classifier.fit(X_train, y_train)

# Step 4: Make predictions on the training data
X_train_prediction = classifier.predict(X_train)

# Calculate training data accuracy
training_data_accuracy = accuracy_score(X_train_prediction, y_train)
print("Training data accuracy:", training_data_accuracy)

#MAking Predective system
input_data = (1.99,494.08,1.99)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = classifier.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The person is fertile')
else:
  print('The person is infertile')


### Creating pickle file

import pickle

filename = 'pcos_prediction model.sav'
pickle.dump(classifier, open(filename,'wb'))

loaded_model = pickle.load(open('pcos_prediction model.sav', 'rb'))

input_data = (1.99,494.08,1.99)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The person is fertile')
else:
  print('The person is infertile')
