from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn import metrics
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import  KNeighborsClassifier
import warnings
import pickle
warnings.filterwarnings('ignore')

# Reading dataset
df = pd.read_csv(r"soil.csv")

df.shape

df.info()

df.describe()

df.head()

df.tail()

# for null values
df.isnull().sum()

df_numeric = df.select_dtypes(include=[np.number])
numeric_cols = df_numeric.columns.values

df_non_numeric = df.select_dtypes(exclude=[np.number])
non_numeric_cols = df_non_numeric.columns.values

#Pandas df.loc attribute access a group of rows and columns by label(s) or a boolean array in the given DataFrame.
df.loc[(df['ph'] < 3.0), 'Soil_Nature'] = 'Ultra acidic'
df.loc[(df['ph'] >= 3.5) & (df['ph'] <= 4.4), 'Soil_Nature'] = 'Extremely acidic'  
df.loc[(df['ph'] >= 4.5) & (df['ph'] <= 5.0), 'Soil_Nature'] = 'Very strongly acidic' 
df.loc[(df['ph'] >= 5.1) & (df['ph'] <= 5.5), 'Soil_Nature'] = 'Strongly acidic'    
df.loc[(df['ph'] >= 5.6) & (df['ph'] <= 6.0), 'Soil_Nature'] = 'Moderately acidic'  
df.loc[(df['ph'] >= 6.1) & (df['ph'] <= 6.5), 'Soil_Nature'] = 'Slightly acidic'  
df.loc[(df['ph'] >= 6.6) & (df['ph'] <= 7.3), 'Soil_Nature'] = 'Neutral' 
df.loc[(df['ph'] >= 7.4) & (df['ph'] <= 7.8), 'Soil_Nature'] = 'Slightly alkaline'  
df.loc[(df['ph'] >= 7.9) & (df['ph'] <= 8.4), 'Soil_Nature'] = 'Moderately alkaline'
df.loc[(df['ph'] >= 8.5) & (df['ph'] <= 9.0), 'Soil_Nature'] = 'Strongly alkaline'  
df.loc[(df['ph'] > 9.0), 'Soil_Nature']  = 'Very strongly alkaline'

# for null values.
df.isnull().sum()

df.dropna(subset = ["Soil_Nature"], inplace=True)

# for null values.
df.isnull().sum()

#Seperating features and target label
features = df[['N','P','K','ph']]
target = df['Soil_Nature']

# Initializing empty lists to append all model's name and corresponding name
acc = []
model = []

# Splitting into train and test data
from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(features,target,test_size = 0.4,random_state =2)

NaiveBayes = GaussianNB()
NaiveBayes.fit(Xtrain,Ytrain)
predicted_values = NaiveBayes.predict(Xtest)
x = metrics.accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('Naive Bayes')
print("Naive Bayes's Accuracy is: ", x*100)
print(classification_report(Ytest,predicted_values))

# Create Base Learners
from sklearn.ensemble import StackingClassifier
estimators = [
     ('NaiveBayes', NaiveBayes)
]

stackclass = StackingClassifier(
     estimators=estimators, final_estimator=SVC(probability=True)
)
stackclass.fit(Xtrain, Ytrain).score(Xtest, Ytest)

filename = 'soilclass.pkl'
soilclass_pkl = open(filename, 'wb')
pickle.dump(stackclass, soilclass_pkl)
soilclass_pkl.close()

#Seperating features and target label for crop prediction
feature = df[['temperature', 'humidity', 'rainfall']]
t = df['Crop']
Xtrain, Xtest, ytrain, ytest = train_test_split(feature,t, test_size = 0.2, random_state =None)
RF = RandomForestClassifier(n_estimators=100, random_state=0)
RF.fit(Xtrain,ytrain)

filename = 'crop.pkl'
crop_pkl = open(filename, 'wb')
pickle.dump(RF, crop_pkl)
crop_pkl.close()