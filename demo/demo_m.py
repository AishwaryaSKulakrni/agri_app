from __future__ import print_function
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import StackingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import  KNeighborsClassifier
import pickle
import warnings
warnings.filterwarnings('ignore')

# Reading soil and crop dataset
df = pd.read_csv(r'soil.csv')

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

# After updating dataset drop na 
df.dropna(subset = ["Soil_Nature"], inplace=True)
df = df.replace(r'^\s*$', np.nan, regex=True)

df.to_csv('soil.csv',index=False)
#Seperating features and target label
soil_feature = df[['N','P','K','ph']]
soil_target = df['Soil_Nature']

# Splitting into train and test data
Xtrain, Xtest, Ytrain, Ytest = train_test_split(soil_feature,soil_target,test_size = 0.4,random_state =2)

# build base estimator for ensemble staking algorithm
NaiveBayes = GaussianNB()
NaiveBayes.fit(Xtrain,Ytrain)

# Create Base Learners
estimators = [
     ('NaiveBayes', NaiveBayes)
]

# Initialize Stacking Classifier with the Meta Learner
soil = StackingClassifier(
     estimators=estimators, final_estimator=SVC()
)
soil.fit(Xtrain, Ytrain)

filename = 'soil.pkl'
soil_pkl = open(filename, 'wb')
pickle.dump(soil, soil_pkl)
soil_pkl.close()

df['Soil_Nature'] = df['Soil_Nature'].astype('category')
df['Soil_Nature_cat'] = df['Soil_Nature'].cat.codes

#Seperating features and target label for crop prediction
crop_feature = df[['temperature', 'humidity', 'rainfall','Soil_Nature_cat']]
crop_target = df['Crop']
# Splitting into train and test data
Xtrain, Xtest, ytrain, ytest = train_test_split(crop_feature,crop_target, test_size = 0.2, random_state =None)
# build crop prediction model
RF = RandomForestClassifier(n_estimators=100, random_state=0)
RF.fit(Xtrain,ytrain)

filename = 'crop.pkl'
crop_pkl = open(filename, 'wb')
pickle.dump(RF, crop_pkl)
crop_pkl.close()

# Reading fertilizer dataset
fert=pd.read_csv(r'Fertilizer Prediction.csv')
fert["Crop"] = fert["Crop"].astype('category')
fert["Crop_cat"] = fert["Crop"].cat.codes

#Seperating features and target label
fert_features = fert[['N','K','P','Crop_cat']]
fert_target = fert['Fertilizer']
# Splitting into train and test data
Xtrain, Xtest, Ytrain, Ytest = train_test_split(fert_features,fert_target,test_size = 0.5,random_state =2)
error_rate = []
for i in range(1, 50):
#pipeline consisting of two stages. The first scales the features, and the second trains a classifier on the resulting dataset
    pipeline = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors = i))
    pipeline.fit(Xtrain, Ytrain)

filename = 'fertilizer.pkl'
fertilizer_pkl = open(filename, 'wb')
pickle.dump(pipeline, fertilizer_pkl)
fertilizer_pkl.close()  