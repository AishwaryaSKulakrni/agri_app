from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn import metrics
from sklearn import tree
import warnings
warnings.filterwarnings('ignore')
import streamlit as st

st.title('Soil Classification, Crop Prediction and Fertilizer recommendation using Deep Learning Apporach')

# Reading dataset
df = pd.read_csv(r"soil.csv")

st.header("Soil Dataset")
st.write(df)